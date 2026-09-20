"""One finite, immutable evaluator run; completion returns to Astra without promotion.

The detached supervisor measures actual progress and resources. Its exclusive lease is
inherited by each stage, so losing the supervisor cannot start a second writer.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import signal
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

from .evaluator_data import atomic, digest, encoded, generate, prepare
from .labeling.process_identity import read_process_identity

ROOT = Path(__file__).resolve().parents[2]
MODULE = "open_shogi_training.evaluator_run"
PAUSED_EXIT = 75
RESOURCE_EXIT = 76
SCHEMA = "open_shogiai_evaluator_run/v1"
STAGES = ("generate", "prepare", "train", "audit", "arena")
ALL_STAGES = (*STAGES, "integrate")


def _stages(config):
    return ALL_STAGES if config.get("development_integration") else STAGES


STATES = {
    "prepared",
    "ready_for_luna",
    "running",
    "stopped",
    "needs_astra",
    "awaiting_astra_review",
    "awaiting_astra_browser",
}
RUNTIME_SOURCES = {
    "replay": "target/release/examples/position_audit",
    "probe": "target/release/examples/core_probe",
    "module": "target/pure/bindings/open_shogi_wasm.js",
    "wasm": "target/pure/bindings/open_shogi_wasm_bg.wasm",
}
# Reviewed operating policy; never changes teacher, sampling or learning parameters.
RESOURCE_POLICY = {
    "revision": "macos-attempt-v1",
    "interval_seconds": 15,
    "maximum_sample_age_seconds": 25,
    "admission_samples": 3,
    "maximum_wait_samples": 5,
    "resume_memory_free_percent": 55,
    "critical_memory_free_percent": 20,
    "resume_disk_margin_gib": 2,
    "resume_rss_fraction": 0.8,
    "quiet_bytes_per_second": 1024**2,
    "swapout_bytes_per_second": 8 * 1024**2,
    "sustained_samples": 2,
    "cooperative_stop_seconds": 80,
}
CALENDAR_POLICY = {
    "revision": "finite-work-v1",
    "run_id": "defense-20260912-recovery-r3",
    "authorization": "explicit_user_calendar_limit_removal_20260915",
    "calendar_wall_limit_seconds": None,
    "historical_active_seconds": None,
    "accounting": "retain task and evaluation consumption; historical total active time unknown",
}
ADMISSION_POLICY = {
    "revision": "itemwise-focus-v1",
    "run_id": "defense-20260912-recovery-r3",
    "authorization": "explicit_user_existing_data_learning_20260917",
    "generation_budget": "closed_consumed_by_authorization; no further generation or hard queue",
    "missing_focus": "exclude dependent observations before loss; retain independent valid targets",
    "memory": "host pressure and swap are diagnostic; allocation failure uses bounded fallback",
    "microbatch": 32,
    "effective_batch": "unchanged; sum losses divided by effective valid examples",
}
ADMISSION_FILES = {
    "training/open_shogi_training/evaluator_run.py",
    "training/open_shogi_training/evaluator_data.py",
    "training/open_shogi_training/evaluator_coverage.py",
    "training/open_shogi_training/defense_scenarios.py",
    "training/open_shogi_training/evaluator_training.py",
    "configs/evaluator-main.json",
}
OPERATIONAL_FILES = {
    "training/open_shogi_training/evaluator_run.py",
    "training/open_shogi_training/evaluator_development.py",
    "configs/evaluator-main.json",
}
POST_TRAINING_FILES = ADMISSION_FILES | {
    "training/open_shogi_training/defense_evaluation.py",
    "training/open_shogi_training/evaluator_arena.py",
    "scripts/compare_evaluator_opponent.py",
}
POST_TRAINING_POLICY = {
    "revision": "optional-screen-v1",
    "run_id": "defense-20260912-recovery-r3",
    "authorization": "explicit_user_post_training_and_development_candidate_20260917",
    "screen": "retained_only",
    "training": "complete; no regeneration, preparation or retraining",
    "evaluation": "unchanged fixed r3 schedule; no automatic promotion",
}


def inside(path: str | Path, *, exists: bool = True) -> Path:
    value = Path(path)
    value = value if value.is_absolute() else ROOT / value
    if not value.is_relative_to(ROOT):
        raise ValueError("path outside repository")
    for parent in (value, *value.parents):
        if parent == ROOT.parent:
            break
        if parent.is_symlink():
            raise ValueError("run paths must not traverse symlinks")
    resolved = value.resolve(strict=exists)
    if not resolved.is_relative_to(ROOT):
        raise ValueError("resolved path outside repository")
    return resolved


def _json(path: Path) -> dict:
    path = inside(path)
    if not path.is_file() or not 0 < path.stat().st_size <= 8 * 1024 * 1024:
        raise ValueError("invalid bounded JSON file")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("JSON contract must be an object")
    # Also rejects Infinity produced by an oversized JSON exponent.
    encoded(value)
    return value


def _number(value, low: float, high: float, name: str, *, integer: bool = True) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not low <= value <= high
        or (integer and type(value) is not int)
    ):
        raise ValueError(f"invalid bounded {name}")


def _hash(value) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None:
        raise ValueError("invalid artifact hash")
    return value


def _validate_config(config: dict) -> None:
    encoded(config)
    if config.get("schema") != SCHEMA:
        raise ValueError("unknown run contract")
    if not isinstance(config.get("run_id"), str) or not re.fullmatch(
        r"[a-zA-Z0-9._-]{1,96}", config["run_id"]
    ):
        raise ValueError("invalid run ID")
    output = inside(config["output"], exists=False)
    if not output.is_relative_to(ROOT / "local" / "runs"):
        raise ValueError("run output must remain under local/runs")
    _number(config["seed"], 0, 2**63 - 1, "seed")
    generation = config["generation"]
    if "round" in config:
        if (config["round"].get("id"), config["round"].get("authorization")) not in {
            ("R4-C1", "explicit_user_20260918"),
            ("R4-C2", "explicit_user_20260920"),
            ("R4-C3", "explicit_user_20260921"),
        }:
            raise ValueError("unapproved learning round")
        ref = generation["prepared_dataset"]
        _hash(ref["manifest_sha256"])
        inside(ref["path"])
        if (
            config["round"]["id"] == "R4-C1"
            and config["evaluation"].get("optional_screen") != "retained_only"
        ):
            raise ValueError("R4-C1 optional screen is retained-only")
    for name, low, high in (
        ("first_game", 0, 10**9),
        ("games", 1, 65536),
        ("workers", 1, 8),
        ("max_plies", 1, 512),
        ("sample_stride", 1, 512),
        ("deviation_stride", 1, 512),
        ("teacher_nodes", 1, 10_000_000),
    ):
        _number(generation[name], low, high, name)
    _hash(generation["leaf_sha256"])
    _hash(generation["teacher_binary_sha256"])
    if generation.get("development_exclusions_path"):
        _hash(generation["development_exclusions_sha256"])
    policy = generation.get("recovery_policy")
    if policy:
        if policy.get("schema") != "open_shogiai_label_recovery/v1":
            raise ValueError("unknown label recovery contract")
        for name, expected in (
            ("main_nodes", 2_000_000),
            ("hard_nodes", 32_000_000),
            ("maximum_attempts_per_task", 2),
            ("maximum_task_nodes", 34_000_000),
            ("maximum_search_seconds", 60),
            ("ready_seconds", 5),
            ("stop_seconds", 1),
            ("quit_seconds", 1),
            ("startup_seconds", 10),
            ("maximum_attempt_seconds", 75),
            ("maximum_task_seconds", 150),
            ("maximum_worker_restarts", 3),
            ("maximum_supplemental_games", 192),
            ("maximum_supplemental_per_family", 16),
        ):
            if policy.get(name) != expected:
                raise ValueError(f"unreviewed recovery budget: {name}")
        for name, expected in (
            ("maximum_hard_attempts", 8192),
            ("maximum_hard_seconds", 43200),
            ("inherited_hard_attempts", 359),
            ("inherited_hard_seconds", 2205.209),
        ):
            if policy.get(name) != expected:
                raise ValueError(f"unreviewed shared hard-queue budget: {name}")
        if policy.get("unique_data_goal") != config["unique_data_goal"]:
            raise ValueError("recovery coverage cannot weaken the learning data goal")
    training = config["training"]
    if config.get("round", {}).get("id") == "R4-C1" and training.get("source_fractions") != [
        0.75,
        0.25,
    ]:
        raise ValueError("unreviewed R4 replay/new mixture")
    if config.get("round", {}).get("id") == "R4-C2" and (
        training.get("source_fractions") != [0.25, 0.75]
        or training.get("coverage_sampler")
        != {
            "epoch_examples": 262144,
            "maximum_per_example": [3, 2],
            "maximum_per_sequence": 512,
            "maximum_per_sequence_batch": 4,
            "minimum_coverage_before_patience": [0.5, 0.6],
        }
        or config.get("development_integration", {}).get("port") != 5175
        or config["evaluation"].get("optional_screen") != "execute"
        or config["evaluation"].get("screen_per_group") != 2
    ):
        raise ValueError("unreviewed R4-C2 coverage/integration policy")
    if training["device"] != "cpu":
        raise ValueError("only the measured CPU training route is enabled")
    for name, maximum in (
        ("threads", 16),
        ("batch_size", 4096),
        ("max_steps", 1_000_000),
        ("max_epochs", 1024),
        ("validation_every", 1_000_000),
        ("patience", 1000),
        ("warmup_steps", 1_000_000),
    ):
        _number(training[name], 1, maximum, name)
    _number(training["learning_rate"], 1e-12, 1, "learning_rate", integer=False)
    _number(
        training["minimum_learning_rate"],
        0,
        training["learning_rate"],
        "minimum_learning_rate",
        integer=False,
    )
    _number(
        training["minimum_relative_improvement"],
        0,
        1,
        "minimum_relative_improvement",
        integer=False,
    )
    _number(training["gradient_norm_limit"], 1e-12, 1e6, "gradient_norm_limit", integer=False)
    if training.get("trained_candidate") not in (None, "best_updated_objective_v1"):
        raise ValueError("unknown trained candidate selection")
    _number(training.get("pair_weight", 0), 0, 1, "pair_weight", integer=False)
    resources = config["resources"]
    for name, low, high, integer in (
        ("maximum_wall_seconds", 1, 604800, True),
        ("free_space_floor_gib", 1, 1024, False),
        ("maximum_process_rss_gib", 1, 20, False),
        ("maximum_swap_growth_gib", 0, 16, False),
        ("stalled_seconds", 1, 86400, True),
        ("maximum_retries", 0, 3, True),
        ("monitor_interval_seconds", 0.1, 60, False),
    ):
        _number(resources[name], low, high, name, integer=integer)
    epoch = config.get("resource_epoch")
    if epoch:
        if (
            not policy
            or epoch.get("authorization") != "explicit_user_approval_20260912"
            or resources["maximum_swap_growth_gib"] != 0.5
            or resources.get("minimum_memory_free_percent") != 50
        ):
            raise ValueError("resource epoch requires the explicitly approved narrow limits")
        for key in ("initial_swap_bytes", "original_initial_swap_bytes"):
            _number(epoch[key], 0, 24 * 1024**3, key)
        _hash(epoch["prior_probe_sha256"])
    for name, expected in (
        ("depth", 64),
        ("max_plies", 256),
        ("paired_3minute_games", 24),
        ("paired_10minute_games", 8),
        ("startpos_demonstration_games", 4 if generation.get("defense_campaign") else 8),
    ):
        if config["evaluation"][name] != expected:
            raise ValueError("evaluation must retain the reviewed finite plan")
    if generation.get("defense_campaign"):
        from .evaluator_training import GROUPS

        if training.get("sampling_fractions") != (
            [0.20, 0.40, 0.20, 0.20]
            if config.get("round", {}).get("id") == "R4-C3"
            else [0.25, 0.30, 0.20, 0.25]
            if "round" in config
            else [0.3, 0.2, 0.3, 0.2]
        ):
            raise ValueError("defense run requires the reviewed four-group sampling plan")
        if training.get("maximum_replay_regression_ratio") != 1.03:
            raise ValueError("unreviewed replay regression guard")
        if config["evaluation"].get("groups") != list(GROUPS):
            raise ValueError("missing defense/attack evaluation groups")
        machine = config["state_machine"]
        if machine.get("schema") != "open_shogiai_evaluator_state/v2" or set(
            machine.get("states", [])
        ) != {"ready_for_luna", "running", "stopped", "needs_astra", "awaiting_astra_review"}:
            raise ValueError("unknown defense state machine")
        if (
            machine.get("transitions")
            != {
                "ready_for_luna": ["running"],
                "running": (
                    ["ready_for_luna"]
                    if generation.get("recovery_policy") or "round" in config
                    else []
                )
                + ["stopped", "needs_astra", "awaiting_astra_review"],
                "stopped": ["running"],
                "needs_astra": [],
                "awaiting_astra_review": [],
            }
            or machine.get("terminal") != "awaiting_astra_review"
        ):
            raise ValueError("unknown defense state transitions")
    exclusions = config["excluded_development_sfens"]
    if (
        not isinstance(exclusions, list)
        or len(exclusions) > 2048
        or any(not isinstance(sfen, str) or not 1 <= len(sfen) <= 512 for sfen in exclusions)
    ):
        raise ValueError("invalid development regression exclusions")


def _code_identity() -> dict:
    scope = [
        "engine",
        "training",
        "scripts",
        "Cargo.lock",
        "pyproject.toml",
        "uv.lock",
        "configs/runtime",
        "configs/evaluator-main.json",
    ]
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *scope],
        cwd=ROOT,
        text=True,
    )
    if dirty:
        raise ValueError("commit the stable run code/config, including new files, before sealing")
    paths = subprocess.check_output(["git", "ls-files", "--", *scope], cwd=ROOT, text=True)
    files = {name: digest(inside(name)) for name in paths.splitlines()}
    if str(Path(__file__).resolve().relative_to(ROOT)) not in files:
        raise ValueError("the actual runner must belong to the sealed code commit")
    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "files": files,
    }


def _reference(path: Path, expected: str | None = None) -> dict:
    path = inside(path)
    if not path.is_file():
        raise ValueError("artifact must be a regular file")
    sha = digest(path)
    if expected is not None and sha != _hash(expected):
        raise ValueError(f"artifact changed: {path.relative_to(ROOT)}")
    return {"path": str(path.relative_to(ROOT)), "sha256": sha}


def _operation_code(run: Path, config: dict, revision: dict | None) -> dict:
    """An execution revision never edits the original experiment or its seal."""
    if revision is None:
        ref = _state(run).get("operation_revision")
        if ref:
            path = inside(ref["path"])
            if path.parent != run / "operations":
                raise ValueError("operation revision outside run")
            _reference(path, ref["sha256"])
            revision = _json(path)
    if revision is None:
        return config["code"]["files"]
    if (
        revision.get("schema") != "open_shogiai_operation_revision/v1"
        or revision.get("run_sha256") != digest(run / "run.json")
        or revision.get("original_code_commit") != config["code"]["commit"]
        or not set(revision["files"])
        <= (
            POST_TRAINING_FILES
            if "post_training" in revision
            else ADMISSION_FILES
            if "dataset_admission" in revision
            else OPERATIONAL_FILES
            | ({"scripts/check_provenance.sh"} if "round" in config else set())
        )
        or not re.fullmatch(r"[a-f0-9]{40}", revision["commit"])
    ):
        raise ValueError("invalid operational code revision")
    if "dataset_admission" in revision:
        if (
            revision.get("admission_policy") != ADMISSION_POLICY
            or config["run_id"] != ADMISSION_POLICY["run_id"]
        ):
            raise ValueError("unrecognized dataset admission revision")
        ref = revision["dataset_admission"]
        if inside(ref["path"]) != run / "data/admission.json":
            raise ValueError("dataset admission outside run")
        _reference(run / "data/admission.json", ref["sha256"])
        admission = _json(run / "data/admission.json")
        if admission["run_sha256"] != digest(run / "run.json") or admission[
            "seal_sha256"
        ] != digest(run / "seal.json"):
            raise ValueError("dataset admission seal changed")
        for ref in admission["evidence"].values():
            _reference(inside(ref["path"]), ref["sha256"])
        config["_dataset_admission"] = revision["dataset_admission"]
    if "post_training" in revision:
        review = revision["post_training"]
        if (
            review.get("policy") != POST_TRAINING_POLICY
            or config["run_id"] != POST_TRAINING_POLICY["run_id"]
        ):
            raise ValueError("unrecognized post-training review")
        for ref in review["evidence"].values():
            _reference(inside(ref["path"]), ref["sha256"])
        config["_post_training"] = review
        config["_optional_screen"] = review["policy"]["screen"]
    if "resource_policy" in revision:
        if revision["resource_policy"] != RESOURCE_POLICY:
            raise ValueError("unrecognized resource policy revision")
        config["_resource_policy"] = revision["resource_policy"]
    if "calendar_policy" in revision:
        if (
            revision["calendar_policy"] != CALENDAR_POLICY
            or config["run_id"] != CALENDAR_POLICY["run_id"]
        ):
            raise ValueError("unrecognized calendar policy or unauthorized run")
        _reference(run / "seal.json", revision["original_seal"]["sha256"])
        if revision["original_maximum_wall_seconds"] != config["resources"]["maximum_wall_seconds"]:
            raise ValueError("original calendar budget changed")
        previous = revision.get("supersedes")
        if previous:
            path = inside(previous["path"])
            if path.parent != run / "operations":
                raise ValueError("previous operation outside run")
            _reference(path, previous["sha256"])
        config["_calendar_policy"] = revision["calendar_policy"]
    if "development_verification" in revision:
        browser = revision["development_verification"]
        name = "scripts/verify-development-candidate.mjs"
        if (
            config.get("round", {}).get("id") != "R4-C3"
            or browser.get("path") != name
            or browser.get("original_sha256")
            != config["development_integration"]["ui_identity"]["files"][name]
            or not re.fullmatch(r"[a-f0-9]{40}", browser.get("commit", ""))
        ):
            raise ValueError("only the C3 browser verification script can be revised")
        committed = subprocess.check_output(
            ["git", "show", f"{browser['commit']}:{name}"], cwd=ROOT.parent / "OpenShogiUI"
        )
        if hashlib.sha256(committed).hexdigest() != _hash(browser["sha256"]):
            raise ValueError("browser revision is not committed code")
        config["development_integration"]["ui_identity"]["files"][name] = browser["sha256"]
    expected = {**config["code"]["files"], **revision["files"]}
    for name, sha in revision["files"].items():
        committed = subprocess.check_output(
            ["git", "show", f"{revision['commit']}:{name}"], cwd=ROOT
        )
        if hashlib.sha256(committed).hexdigest() != sha:
            raise ValueError("operation revision is not committed code")
        if name == "scripts/check_provenance.sh":
            original = subprocess.check_output(
                ["git", "show", f"{config['code']['commit']}:{name}"], cwd=ROOT
            )
            binding = subprocess.check_output(
                [
                    "git",
                    "show",
                    f"{config['code']['commit']}:bindings/wasm/open_shogi_wasm_bg.wasm",
                ],
                cwd=ROOT,
            )
            pin = rb'("open_shogi_wasm_bg.wasm": ")[a-f0-9]{64}(")'
            expected_script, count = re.subn(
                pin, rb"\g<1>" + hashlib.sha256(binding).hexdigest().encode() + rb"\g<2>", original
            )
            if count != 1 or committed != expected_script:
                raise ValueError(
                    "provenance revision may only correct the original committed binding pin"
                )
        if name == "configs/evaluator-main.json":
            original = subprocess.check_output(
                ["git", "show", f"{config['code']['commit']}:{name}"], cwd=ROOT
            )
            if hashlib.sha256(original).hexdigest() != config["code"]["files"][name]:
                raise ValueError("original source config identity changed")
            old, new = json.loads(original), json.loads(committed)
            old.pop("operations", None)
            new.pop("operations", None)
            if old != new:
                raise ValueError("operational resume cannot change experiment conditions")
    return expected


def verify(run: Path, *, operation_revision: dict | None = None) -> dict:
    run = inside(run)
    receipt = _json(run / "seal.json")
    if receipt.get("schema") != "open_shogiai_evaluator_seal/v1":
        raise ValueError("missing or unknown seal receipt")
    if digest(inside(run / "run.json")) != _hash(receipt["run_sha256"]):
        raise ValueError("sealed run.json changed; preserve assets and create a new run")
    config = _json(run / "run.json")
    _validate_config(config)
    if inside(config["output"]) != run or receipt["run_id"] != config["run_id"]:
        raise ValueError("sealed run location/identity mismatch")
    if config.get("execution_cwd", str(ROOT)) != str(ROOT):
        raise ValueError("run belongs to another execution root")
    if receipt["code_commit"] != config["code"]["commit"]:
        raise ValueError("sealed commit identity mismatch")
    code_files = _operation_code(run, config, operation_revision)
    for name, expected in code_files.items():
        _reference(inside(name), expected)
    if set(config["runtime"]) != set(RUNTIME_SOURCES):
        raise ValueError("incomplete runtime snapshot")
    for ref in config["runtime"].values():
        path = inside(ref["path"])
        if path.parent != run / "runtime":
            raise ValueError("runtime must be copied inside this run")
        _reference(path, ref["sha256"])
    for key, ref in config["inputs"].items():
        expected = ref["sha256"]
        if key == "source_config" and ref["path"] == "configs/evaluator-main.json":
            expected = code_files.get(ref["path"], expected)
        _reference(inside(ref["path"]), expected)
    if "round" in config:
        config["_resource_policy"] = RESOURCE_POLICY
        config["_optional_screen"] = config["evaluation"]["optional_screen"]
    return config


def _state(run: Path) -> dict:
    value = _json(run / "state.json")
    if value.get("status") not in STATES:
        raise ValueError("unknown run state")
    seal = _json(run / "seal.json")
    config = _json(run / "run.json")
    if (
        config["generation"].get("defense_campaign")
        and value.get("status") not in config["state_machine"]["states"]
    ):
        raise ValueError("state is outside this defense contract")
    if value.get("run_id") != seal["run_id"] or (
        config["generation"].get("defense_campaign")
        and (
            value.get("run_sha256") != seal["run_sha256"]
            or value.get("schema") != "open_shogiai_evaluator_state/v2"
        )
    ):
        raise ValueError("run state belongs to another contract")
    if value["status"] == "awaiting_astra_review":
        review = _json(run / "candidate-review.json")
        _reference(run / "candidate-review.json", value["review_sha256"])
        for key, path in (
            ("run_sha256", run / "run.json"),
            ("candidate_sha256", selected_model(run)),
            ("offline_sha256", run / "development-test.json"),
            ("arena_sha256", run / "arena/arena.json"),
        ):
            _reference(path, review[key])
    return value


def seal(config_path: Path) -> dict:
    config_path = inside(config_path)
    config = _json(config_path)
    _validate_config(config)
    code = _code_identity()
    run = inside(config["output"], exists=False)
    if run.exists():
        raise ValueError("run already exists; use status/start to resume its immutable contract")
    generation = config["generation"]
    inputs = {"source_config": _reference(config_path)}
    inherited = {}
    parent = (
        config.get("recovery_from")
        if generation.get("defense_campaign") and generation.get("recovery_policy")
        else None
    )
    if parent:
        parent_run = inside(parent["path"])
        with _lease(parent_run):
            parent_state = _state(parent_run)
            if status(parent_run)["process_alive"] or _residual_stage_group(parent_run) is not None:
                raise ValueError("parent run still active; recovery requires a stopped writer")
            if parent_state["status"] != "needs_astra" or parent_state.get("stage") != "generate":
                raise ValueError("recovery requires the reviewed stopped generation stage")
            if (parent_run / "fit").exists():
                raise ValueError("generation recovery cannot migrate training state")
            inputs["parent_contract"] = _reference(parent_run / "run.json", parent["run_sha256"])
            inputs["parent_state"] = _reference(parent_run / "state.json", parent["state_sha256"])
            inputs["parent_inventory"] = _reference(
                inside(parent["inventory_path"]), parent["inventory_sha256"]
            )
            inputs["parent_compatibility"] = _reference(
                inside(parent["compatibility_path"]), parent["compatibility_sha256"]
            )
            hard_evidence = generation["recovery_policy"]["hard_budget_evidence"]
            inputs["parent_hard_budget"] = _reference(
                inside(hard_evidence["path"]), hard_evidence["sha256"]
            )
            parent_config = _json(parent_run / "run.json")
            if parent_config["run_id"] != parent["run_id"]:
                raise ValueError("parent run ID mismatch")
            inherited = {
                key: parent_state[key] for key in ("began_at", "initial_swap_bytes", "retries")
            }
            if config.get("resource_epoch"):
                epoch = config["resource_epoch"]
                inputs["prior_resource_stop"] = _reference(
                    inside(epoch["prior_probe"]), epoch["prior_probe_sha256"]
                )
                inherited = _apply_resource_epoch(inherited, epoch)
            inherited["parent_run_id"] = parent["run_id"]
            inherited["parent_state_sha256"] = parent["state_sha256"]

    for name, path_key, hash_key in (
        ("leaf", "leaf_path", "leaf_sha256"),
        ("teacher_config", "teacher_config_path", "teacher_config_sha256"),
        ("split_guard", "split_guard_path", "split_guard_sha256"),
        ("development_exclusions", "development_exclusions_path", "development_exclusions_sha256"),
    ):
        if path_key not in generation:
            if name == "development_exclusions":
                continue
            raise ValueError(f"missing {path_key}")
        inputs[name] = _reference(inside(generation[path_key]), generation.get(hash_key))
        generation[hash_key] = inputs[name]["sha256"]
    if generation.get("prepared_dataset"):
        from .r4_data import verify_dataset

        source = inside(generation["prepared_dataset"]["path"])
        manifest = verify_dataset(ROOT, source, generation["prepared_dataset"]["manifest_sha256"])
        inputs["prepared_manifest"] = _reference(source / "manifest.json")
        for i, ref in enumerate(manifest["artifacts"]):
            inputs[f"prepared_{i}"] = _reference(source / ref["path"], ref["sha256"])
    for i, ref in enumerate(config.get("data_preparation", {}).get("license_evidence", [])):
        inputs[f"source_rights_{i}"] = _reference(inside(ref["path"]), ref["sha256"])
    sources = {name: _reference(inside(path)) for name, path in RUNTIME_SOURCES.items()}
    campaign = generation.get("defense_campaign")
    if campaign:
        from .labeling.config import load_teacher_config

        installed_teacher = load_teacher_config(inside(generation["teacher_config_path"]))
        if installed_teacher.install_manifest is None:
            raise ValueError("teacher installation provenance is required")
        inputs["teacher_install_manifest"] = _reference(inside(installed_teacher.install_manifest))
        inputs["regression_positions"] = _reference(
            inside(config["evaluation"]["regression_positions_path"]),
            config["evaluation"]["regression_positions_sha256"],
        )
        teacher_identity = _json(inside("local/frozen/metadata/teacher-identity.json"))
        inputs["teacher_identity"] = _reference(
            inside("local/frozen/metadata/teacher-identity.json")
        )
        for i, ref in enumerate([teacher_identity["binary"], *teacher_identity["eval_files"]]):
            inputs[f"teacher_asset_{i}"] = _reference(inside(ref["path"]), ref["sha256"])
        replay = campaign["replay_dataset"]
        inputs["replay_manifest"] = _reference(
            inside(Path(replay["path"]) / "manifest.json"), replay["manifest_sha256"]
        )
        manifest = _json(inside(inputs["replay_manifest"]["path"]))
        for i, ref in enumerate(manifest["artifacts"]):
            inputs[f"replay_artifact_{i}"] = _reference(
                inside(Path(replay["path"]) / ref["path"]), ref["sha256"]
            )
    if config.get("development_integration"):
        from .evaluator_development import ui_identity

        config["development_integration"]["ui_identity"] = ui_identity(ROOT)
    run.parent.mkdir(parents=True, exist_ok=True)
    # This temporary directory is exclusively created here and never named by a sealed run.
    staging = Path(tempfile.mkdtemp(prefix=f".{run.name}.sealing-", dir=run.parent))
    try:
        (staging / "runtime").mkdir()
        runtime = {}
        for name, ref in sources.items():
            source = inside(ref["path"])
            destination = staging / "runtime" / source.name
            shutil.copy2(source, destination)
            if digest(destination) != ref["sha256"] or digest(source) != ref["sha256"]:
                raise ValueError("runtime changed during snapshot copy")
            runtime[name] = {
                "path": str((run / "runtime" / source.name).relative_to(ROOT)),
                "sha256": ref["sha256"],
            }
        generation.update(
            seed=config["seed"],
            replay_path=runtime["replay"]["path"],
            replay_sha256=runtime["replay"]["sha256"],
        )
        if campaign:
            campaign.update(
                probe_path=runtime["probe"]["path"], probe_sha256=runtime["probe"]["sha256"]
            )
        config["training"].update(
            seed=config["seed"],
            initial_model=generation["leaf_path"],
            initial_sha256=generation["leaf_sha256"],
        )
        config.update(
            code=code,
            runtime=runtime,
            inputs=inputs,
            sealed_at=time.time(),
            execution_cwd=str(ROOT),
        )
        for ref in inputs.values():
            _reference(inside(ref["path"]), ref["sha256"])
        for path, expected in code["files"].items():
            _reference(inside(path), expected)
        atomic(staging / "run.json", encoded(config))
        receipt = {
            "schema": "open_shogiai_evaluator_seal/v1",
            "run_id": config["run_id"],
            "run_sha256": digest(staging / "run.json"),
            "code_commit": code["commit"],
        }
        atomic(staging / "seal.json", encoded(receipt))
        atomic(
            staging / "state.json",
            encoded(
                {
                    "schema": "open_shogiai_evaluator_state/v2"
                    if campaign
                    else "open_shogiai_evaluator_state/v1",
                    "status": "ready_for_luna" if campaign else "prepared",
                    "run_id": config["run_id"],
                    "pid": None,
                    "run_sha256": receipt["run_sha256"],
                    **inherited,
                }
            ),
        )
        if run.exists():
            raise ValueError("run appeared during sealing; nothing was replaced")
        staging.rename(run)
    finally:
        if staging.exists():
            shutil.rmtree(staging)  # Only unpublished files created by this seal attempt.
    return {
        "run_id": config["run_id"],
        "run": str(run.relative_to(ROOT)),
        "status": "ready_for_luna" if campaign else "prepared",
        "code_commit": code["commit"],
        "run_sha256": receipt["run_sha256"],
    }


@contextlib.contextmanager
def _lease(run: Path, inherited_fd: int | None = None):
    path = inside(run / "operation.lock", exists=False)
    if inherited_fd is None:
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    else:
        descriptor = inherited_fd
        if descriptor < 3:
            raise ValueError("lease must not alias standard streams")
        observed, expected = os.fstat(descriptor), path.stat()
        if (observed.st_dev, observed.st_ino) != (expected.st_dev, expected.st_ino):
            raise ValueError("inherited lease belongs to another run")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield descriptor
    finally:
        # Do not LOCK_UN: children inherit the same open-file description.
        os.close(descriptor)


def _launch(run: Path, command_args: list[str], log_name: str, lease: int):
    """Own fresh parent log handles only through spawn; children own dup2 copies.

    A detached macOS PTY may remain in the descriptor table after revocation but
    fail fstat during Python's init_sys_streams. Never inherit runner stdin.
    Only the live lease is passed between this process tree, never saved for resume.
    Teacher/native protocol pipes are separately owned by their existing clients.
    """
    if lease < 3 or Path(log_name).name != log_name:
        raise ValueError("invalid launcher lease/log")
    os.fstat(lease)
    with inside(run / log_name, exists=False).open("ab") as log:
        return subprocess.Popen(
            command_args,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            close_fds=True,
            start_new_session=True,
            pass_fds=(lease,),
        )


def _remember_error(state: dict, phase: str, error: BaseException) -> None:
    state.setdefault("errors", []).append(
        {"phase": phase, "traceback": "".join(traceback.format_exception(error))}
    )


def _save_state(run: Path, state: dict) -> None:
    try:
        atomic(run / "state.json", encoded(state))
    except OSError as error:
        # A diagnostic write failure must not erase the original exception chain.
        primary = "\n".join(item["traceback"] for item in state.get("errors", []))
        if primary:
            raise RuntimeError(
                f"state persistence failed; original failures:\n{primary}"
            ) from error
        raise


def _resume_snapshot(run: Path, config: dict) -> dict:
    """Read durable shards and cursors under the lease; never repair data in place."""
    from .evaluator_data import Replay

    data = run / "data"
    if config["generation"].get("prepared_dataset"):
        snapshot = {"games": 0, "rows": 0, "files": {}, "pending": {}, "tasks": {}, "counters": {}}
        for stage in ("generate", "prepare"):
            if (run / f"{stage}-complete.json").exists():
                _verify_completion(run, stage, config)
                snapshot["files"][f"{stage}-complete.json"] = digest(run / f"{stage}-complete.json")
        if (data / "dataset/manifest.json").exists():
            dataset = _dataset(run, config)
            snapshot["rows"] = sum(dataset["unique_positions"].values())
        if (run / "fit/resume.json").exists():
            ref = _json(run / "fit/resume.json")
            path = inside(run / "fit" / ref["path"])
            if path.parent != run / "fit":
                raise ValueError("checkpoint outside fit")
            _reference(path, ref["sha256"])
            snapshot["checkpoint"] = ref
        return snapshot
    if (data / "generation.json").read_bytes() != encoded(config["generation"]):
        raise ValueError("generation cursor belongs to another configuration")
    inventory, games, rows = {}, set(), 0
    for path in sorted((data / "games").glob("*.receipt.json")):
        path = inside(path)
        saved = _json(path)
        game = saved["game"]
        if game in games or path.name != f"{game:06d}.json.receipt.json":
            raise ValueError("duplicate or misnamed trajectory")
        shard = inside(path.with_name(f"{game:06d}.json.gz"))
        _reference(shard, saved["sha256"])
        raw = json.loads(gzip.decompress(shard.read_bytes()))
        if raw["game"] != game or len(raw["records"]) != saved["rows"]:
            raise ValueError("trajectory/receipt row mismatch")
        games.add(game)
        rows += saved["rows"]
        for item in (shard, path):
            inventory[str(item.relative_to(run))] = digest(item)
    if (data / "inherited.json").exists():
        for saved in _json(data / "inherited.json")["games"]:
            _reference(data / "games" / f"{saved['game']:06d}.json.gz", saved["sha256"])
            _reference(
                data / "games" / f"{saved['game']:06d}.json.receipt.json", saved["receipt_sha256"]
            )
    progress = (
        _json(data / "generation-progress.json")
        if (data / "generation-progress.json").exists()
        else {}
    )
    if len(games) < progress.get("games", 0):
        raise ValueError("committed trajectories disappeared behind progress cursor")
    db_path = inside(data / "tasks.sqlite3")
    with contextlib.closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as db:
        if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise ValueError("invalid recovery ledger")
        checkpoints = dict(db.execute("SELECT game,payload FROM checkpoints"))
        counters = dict(db.execute("SELECT name,value FROM counters"))
        tasks = dict(db.execute("SELECT status,count(*) FROM tasks GROUP BY status"))
        task_rows = list(db.execute("SELECT * FROM tasks ORDER BY id"))
        for key, identity, task_status, attempts, result, _updated in task_rows:
            if hashlib.sha256(identity.encode()).hexdigest() != key:
                raise ValueError("ledger task identity hash mismatch")
            json.loads(attempts)
            if task_status == "accepted" and not isinstance(json.loads(result or "null"), dict):
                raise ValueError("accepted task payload missing")
        ledger_sha = hashlib.sha256(encoded(task_rows))
    pending = {}
    replay = None
    try:
        for game, payload in checkpoints.items():
            value = json.loads(payload)
            if game not in games:
                if value["ply"] != len(value["moves"]):
                    raise ValueError("checkpoint ply/history mismatch")
                if replay is None:
                    replay = Replay(ROOT, config["generation"])
                position = replay.ask(reset=value["initial_sfen"], successors=True)
                for move in value["moves"]:
                    position = replay.ask(movement=move, successors=True)
                if position["sfen"] != value["sfen"]:
                    raise ValueError("checkpoint native replay mismatch")
                pending[str(game)] = {
                    "ply": value["ply"],
                    "rows": len(value["records"]),
                    "sha256": hashlib.sha256(payload.encode()).hexdigest(),
                }
    finally:
        if replay is not None:
            replay.close()
    # Preserve unreceipted output and atomic-write remnants; generation resumes
    # from the transaction cursor and cached accepted tasks, never from temp bytes.
    for path in (data / "games").glob("*"):
        path = inside(path)
        if path.is_file() and str(path.relative_to(run)) not in inventory:
            if path.name.endswith(".json.gz"):
                raw = json.loads(gzip.decompress(path.read_bytes()))
                if raw["game"] not in checkpoints:
                    raise ValueError("uncommitted trajectory lacks durable cursor")
            inventory[str(path.relative_to(run))] = digest(path)
    snapshot = {
        "games": len(games),
        "rows": rows,
        "files": inventory,
        "pending": pending,
        "tasks": tasks,
        "counters": counters,
        "ledger_sha256": ledger_sha.hexdigest(),
    }
    prior = _state(run)
    if prior.get("execution_attempt"):
        prefix = run / "attempts" / f"{prior['execution_attempt']:06d}"
        previous_path = Path(f"{prefix}-pause.json")
        if not previous_path.exists():
            previous_path = prefix.with_suffix(".json")
        previous = _json(previous_path)["snapshot"]
        for name, sha in previous["files"].items():
            if name.endswith(".receipt.json"):
                _reference(run / name, sha)
                shard = name.replace(".receipt.json", ".gz")
                _reference(run / shard, previous["files"][shard])
        if any(
            counters.get(key, -1) < value
            for key, value in previous["counters"].items()
            if key != "hard_seconds"
        ):
            raise ValueError("cumulative execution budget decreased")
        if tasks.get("accepted", 0) < previous["tasks"].get("accepted", 0):
            raise ValueError("accepted task cursor decreased")
    return snapshot


def _swap_recoverable(run: Path, state: dict) -> bool:
    if (
        state.get("reason") != "swap_limit"
        or state.get("stage") != "generate"
        or state.get("errors")
        or state.get("cleanup", {}).get("remaining_processes") != {}
        or state.get("cleanup", {}).get("inventory_errors")
        or not state.get("execution_attempt")
    ):
        return False
    result = _json(run / "attempts" / f"{state['execution_attempt']:06d}-result.json")
    return result == state


def _startup_recoverable(run: Path, state: dict) -> bool:
    # Prepare only derives arrays from sealed receipts; a supervisor-confirmed pause
    # may restart its incomplete staging, never generation or an optimizer update.
    if (
        state.get("stage") == "prepare"
        and state.get("reason") in {"stage_exit_-15_during_stop", "stage_exit_-9_during_stop"}
        and (run / "STOP").exists()
        and state.get("stage_pid") in state.get("cleanup", {}).get("signalled_pids", [])
        and state.get("cleanup", {}).get("remaining_processes") == {}
        and not state.get("cleanup", {}).get("inventory_errors")
        and not state.get("errors")
        and state.get("execution_attempt")
        and _json(run / "attempts" / f"{state['execution_attempt']:06d}-result.json") == state
    ):
        return True
    if _swap_recoverable(run, state):
        return True
    if state.get("startup_failure") == "before_spawn":
        return True
    # Compatibility for the original pre-interpreter failure, which could not
    # write a stage registration or structured failure receipt.
    if state.get("reason") != "stage_exit_1" or state.get("stage") != "generate":
        return False
    receipt = _json(run / "stage-process.json") if (run / "stage-process.json").exists() else {}
    if receipt.get("pid") == state.get("stage_pid"):
        return False
    log = inside(run / f"generate-{state.get('retries', {}).get('generate', 0)}.log")
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size - 4096))
        tail = stream.read()
    return (
        b"Fatal Python error: init_sys_streams: can't initialize sys standard streams\n" in tail
        and b"OSError: [Errno 9] Bad file descriptor\n" in tail
        and tail.rstrip().endswith(b"<no Python frame>")
        and state.get("cleanup", {}).get("remaining_processes") == {}
        and not state.get("errors")
    )


def _reviewed_generation_failure(run: Path, current: dict) -> bool:
    path = run / "data/admission.json"
    if not path.exists():
        return False
    admission = _json(path)
    return (
        admission.get("policy") == ADMISSION_POLICY["revision"]
        and admission.get("original_state_sha256") == hashlib.sha256(encoded(current)).hexdigest()
        and admission.get("run_sha256") == digest(run / "run.json")
        and current.get("status") == "needs_astra"
        and current.get("stage") == "generate"
        and current.get("reason") == "stage_exit_1"
        and not current.get("errors")
    )


def _admit_existing_data(run: Path) -> dict:
    """Explicit scientific admission, not a startup-only or resource revision."""
    path = run / "data/admission.json"
    if path.exists():
        return _reference(path)
    current = _state(run)
    coverage = _json(run / "data/recovery-coverage.json")
    log = run / f"generate-{current.get('retries', {}).get('generate', 0)}.log"
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size - 4096))
        tail = stream.read()
    if (
        _json(run / "run.json")["run_id"] != ADMISSION_POLICY["run_id"]
        or current.get("status") != "needs_astra"
        or current.get("stage") != "generate"
        or current.get("reason") != "stage_exit_1"
        or current.get("errors")
        or current.get("cleanup", {}).get("remaining_processes") != {}
        or current.get("cleanup", {}).get("inventory_errors")
        or coverage["reasons"] != ["focus_total"]
        or not tail.rstrip().endswith(
            b"ValueError: finite generation coverage exhausted: coverage_exhausted"
        )
        or _json(run / "attempts" / f"{current['execution_attempt']:06d}-result.json") != current
    ):
        raise ValueError(
            "existing-data admission requires the reviewed focus-only generation failure"
        )
    for pid, key in (("pid", "process_identity"), ("stage_pid", "stage_identity")):
        if current.get(key) and read_process_identity(current.get(pid)) == current[key]:
            raise ValueError("live process prevents admission")
    if _residual_stage_group(run) is not None:
        raise ValueError("residual group prevents admission")
    config = _json(run / "run.json")
    snapshot = _resume_snapshot(run, config)
    evidence = run / "admission-evidence"
    evidence.mkdir(exist_ok=True)
    refs = {}
    for name, value in {
        "state": current,
        "coverage": coverage,
        "snapshot": snapshot,
        "generation-progress": _json(run / "data/generation-progress.json"),
    }.items():
        target = evidence / f"{name}.json"
        if target.exists() and target.read_bytes() != encoded(value):
            raise ValueError("admission evidence changed")
        atomic(target, encoded(value))
        refs[name] = _reference(target)
    target = evidence / "generation-failure.log"
    atomic(target, log.read_bytes())
    refs["failure_log"] = _reference(target)
    value = {
        "schema": "open_shogiai_dataset_admission/v1",
        "policy": ADMISSION_POLICY["revision"],
        "authorization": ADMISSION_POLICY,
        "run_sha256": digest(run / "run.json"),
        "seal_sha256": digest(run / "seal.json"),
        "generation_sha256": digest(run / "data/generation.json"),
        "original_state_sha256": hashlib.sha256(encoded(current)).hexdigest(),
        "evidence": refs,
        "generation_budget_closed": True,
        "consumption": snapshot["counters"],
        "pending_cursors": len(snapshot["pending"]),
        "tasks": snapshot["tasks"],
    }
    atomic(path, encoded(value))
    return _reference(path)


def _finalize_existing_generation(run: Path, config: dict, snapshot: dict) -> None:
    if "_dataset_admission" not in config or (run / "generate-complete.json").exists():
        return
    from .evaluator_coverage import coverage_report

    admission = _json(run / "data/admission.json")
    original = _json(inside(admission["evidence"]["snapshot"]["path"]))
    if snapshot != original:
        raise ValueError("admitted generation data or cursor changed")
    coverage = coverage_report(run / "data", config["generation"])
    if not coverage["passed"]:
        raise ValueError(
            "non-focus generation coverage remains invalid: " + str(coverage["reasons"])
        )
    result = {
        **_json(inside(admission["evidence"]["generation-progress"]["path"])),
        "status": "complete",
        "coverage": coverage,
        "focus_quality": coverage["focus_quality"],
        "dataset_admission": config["_dataset_admission"],
        "generation_budget_closed": True,
    }
    atomic(run / "data/generation-complete.json", encoded(result))
    atomic(
        run / "generate-complete.json",
        encoded(
            {
                "schema": "open_shogiai_evaluator_stage/v1",
                "stage": "generate",
                "run_sha256": digest(run / "run.json"),
                "result": result,
                "artifacts": _completion_artifacts(run, "generate"),
                "dataset_admission": config["_dataset_admission"],
            }
        ),
    )


def _review_post_training(run: Path) -> dict:
    """Bind the reviewed optional-screen failure to the already completed model."""
    current, config = _state(run), _json(run / "run.json")
    log = run / f"audit-{current.get('retries', {}).get('audit', 0)}.log"
    tail = log.read_bytes()[-16384:]
    if (
        config["run_id"] != POST_TRAINING_POLICY["run_id"]
        or current.get("status") != "needs_astra"
        or current.get("stage") != "audit"
        or current.get("reason") != "stage_exit_1"
        or current.get("errors")
        or current.get("cleanup", {}).get("remaining_processes") != {}
        or current.get("cleanup", {}).get("inventory_errors")
        or b"defense_evaluation.py" not in tail
        or b"USIIncompleteDepthError" not in tail
        or re.search(
            rb"open_shogi_training.evaluator_ledger.DeferredTaskError: [a-f0-9]{64}\s*$", tail
        )
        is None
        or _json(run / "attempts" / f"{current['execution_attempt']:06d}-result.json") != current
    ):
        raise ValueError("post-training review requires the recorded optional depth failure")
    folder = run / "post-training-evidence"
    folder.mkdir(exist_ok=True)
    for name, content in (
        ("state.json", encoded(current)),
        ("audit-failure.log", log.read_bytes()),
    ):
        path = folder / name
        if path.exists() and path.read_bytes() != content:
            raise ValueError("post-training failure evidence changed")
        atomic(path, content)
    return {
        "policy": POST_TRAINING_POLICY,
        "original_state_sha256": hashlib.sha256(encoded(current)).hexdigest(),
        "training_operation": _json(run / "fit/training.json")["identity"]["operation_revision"],
        "evidence": {
            "state": _reference(folder / "state.json"),
            "failure_log": _reference(folder / "audit-failure.log"),
            "training": _reference(run / "train-complete.json"),
            "runtime_audit": _reference(run / "model-audit.json"),
        },
    }


def _reviewed_post_training_failure(run: Path, current: dict, review: dict | None = None) -> bool:
    if review is None and (run / "approved-operation.json").exists():
        ref = _json(run / "approved-operation.json")
        path = inside(ref["path"])
        if path.parent != run / "operations":
            raise ValueError("approved operation outside run")
        _reference(path, ref["sha256"])
        review = _json(path).get("post_training")
    return bool(
        review
        and review.get("policy") == POST_TRAINING_POLICY
        and review.get("original_state_sha256") == hashlib.sha256(encoded(current)).hexdigest()
    )


def _resume_idle_state(run: Path, *, post_training: dict | None = None) -> dict:
    current = _state(run)
    pending = current["status"] == "running" and current.get("reason") == "startup_pending"
    if pending:
        attempt = _json(run / "attempts" / f"{current['execution_attempt']:06d}.json")
        if attempt["operation_revision"] != current.get("operation_revision") or current.get("pid"):
            raise ValueError("invalid interrupted startup attempt")
    if not pending and current["status"] not in {
        "ready_for_luna",
        "stopped",
        "needs_astra",
        "running",
    }:
        raise ValueError("resume requires a paused run or confirmed startup failure")
    if current["status"] == "needs_astra" and not (
        _startup_recoverable(run, current)
        or _reviewed_generation_failure(run, current)
        or _reviewed_post_training_failure(run, current, post_training)
    ):
        raise ValueError("failure is outside startup recovery; Astra review required")
    if current["status"] == "running" and current.get("errors"):
        raise ValueError("interrupted supervisor has unresolved errors")
    for pid_key, identity_key in (("pid", "process_identity"), ("stage_pid", "stage_identity")):
        if (
            current.get(identity_key)
            and read_process_identity(current.get(pid_key)) == current[identity_key]
        ):
            raise ValueError("live process prevents resume")
    if _residual_stage_group(run) is not None:
        raise ValueError("residual stage group prevents resume")
    return current


def _approve_operations(
    run: Path,
    *,
    abolish_calendar_limit: bool = False,
    admit_existing_data: bool = False,
    post_training_evaluation: bool = False,
) -> dict:
    """Astra's explicit one-time code review; resume never approves a new hash."""
    with _lease(run):
        previous_ref, previous = None, {}
        if (run / "approved-operation.json").exists():
            previous_ref = _json(run / "approved-operation.json")
            previous_path = inside(previous_ref["path"])
            if previous_path.parent != run / "operations":
                raise ValueError("approved operation outside run")
            _reference(previous_path, previous_ref["sha256"])
            previous = _json(previous_path)
        admission_ref = _admit_existing_data(run) if admit_existing_data else None
        post_training = previous.get("post_training")
        if post_training_evaluation and post_training is None:
            post_training = _review_post_training(run)
        _resume_idle_state(run, post_training=post_training)
        config, code = _json(run / "run.json"), _code_identity()
        added = set(code["files"]) - set(config["code"]["files"])
        if set(config["code"]["files"]) - set(code["files"]) or (
            added and (not post_training or added != {"scripts/compare_evaluator_opponent.py"})
        ):
            raise ValueError("operational resume cannot add/remove sealed code files")
        changed = {
            name: sha
            for name, sha in code["files"].items()
            if sha != config["code"]["files"].get(name)
        }
        revision = {
            "schema": "open_shogiai_operation_revision/v1",
            "run_sha256": digest(run / "run.json"),
            "original_code_commit": config["code"]["commit"],
            "commit": code["commit"],
            "files": changed,
        }
        if config.get("round", {}).get("id") == "R4-C3":
            from .evaluator_development import ui_identity

            original = config["development_integration"]["ui_identity"]["files"]
            ui = ui_identity(ROOT)
            changed_ui = {
                p
                for p in original.keys() | ui["files"].keys()
                if original.get(p) != ui["files"].get(p)
            }
            name = "scripts/verify-development-candidate.mjs"
            if changed_ui:
                if changed_ui != {name}:
                    raise ValueError("operational revision cannot change OSUI product code")
                revision["development_verification"] = {
                    "path": name,
                    "commit": ui["commit"],
                    "original_sha256": original[name],
                    "sha256": ui["files"][name],
                }
        if config.get("resource_epoch"):
            revision["resource_policy"] = RESOURCE_POLICY
        if abolish_calendar_limit or "calendar_policy" in previous:
            revision.update(
                calendar_policy=CALENDAR_POLICY,
                original_seal=_reference(run / "seal.json"),
                original_maximum_wall_seconds=config["resources"]["maximum_wall_seconds"],
                supersedes=previous_ref,
            )
            # Retrying after pointer publication must reuse the same approval.
            if {k: v for k, v in previous.items() if k != "supersedes"} == {
                k: v for k, v in revision.items() if k != "supersedes"
            }:
                revision = previous
        if admission_ref or "dataset_admission" in previous:
            revision.update(
                dataset_admission=admission_ref or previous["dataset_admission"],
                admission_policy=ADMISSION_POLICY,
            )
            if {k: v for k, v in previous.items() if k != "supersedes"} == {
                k: v for k, v in revision.items() if k != "supersedes"
            }:
                revision = previous
        if post_training or "post_training" in previous:
            revision["post_training"] = post_training or previous["post_training"]
        if {k: v for k, v in previous.items() if k != "supersedes"} == {
            k: v for k, v in revision.items() if k != "supersedes"
        }:
            revision = previous
        config = verify(run, operation_revision=revision)
        _require_migration(run, config)
        _resume_snapshot(run, config)
        if "post_training" in revision:
            for stage in ("generate", "prepare", "train"):
                _verify_completion(run, stage, config)
            _audit_report(run, config)
        path = run / "operations" / f"{hashlib.sha256(encoded(revision)).hexdigest()}.json"
        if not path.exists():
            atomic(path, encoded(revision))
        ref = _reference(path)
        atomic(run / "approved-operation.json", encoded(ref))
        return ref


def _record_resume(run: Path, status: str, reason: str, **details) -> None:
    # Caller owns the writer lease; an old rejection cannot overwrite a newer start.
    atomic(
        run / "last-resume.json",
        encoded({"at": time.time(), "status": status, "reason": reason, **details}),
    )


def _prepare_resume(run: Path) -> tuple[dict, dict]:
    current = _resume_idle_state(run)
    if "round" in _json(run / "run.json") and not (run / "approved-operation.json").exists():
        # A new round is authorized by its original seal. Operational revisions
        # exist only for legacy recoveries; do not manufacture one for each round.
        ref, revision_path = None, None
        config = verify(run)
        expected_code = config["code"]["files"]
    else:
        ref = _json(run / "approved-operation.json")
        revision_path = inside(ref["path"])
        if revision_path.parent != run / "operations":
            raise ValueError("approved operation outside run")
        _reference(revision_path, ref["sha256"])
        revision = _json(revision_path)
        config = verify(run, operation_revision=revision)
        expected_code = {**config["code"]["files"], **revision["files"]}
    code = _code_identity()
    if code["files"] != expected_code:
        raise ValueError("resume code differs from Astra-approved code")
    _require_migration(run, config)
    snapshot = _resume_snapshot(run, config)
    if "round" not in config:
        _finalize_existing_generation(run, config, snapshot)
    for stage in _stages(config):
        if (run / f"{stage}-complete.json").exists():
            _verify_completion(run, stage, config)
    if current.get("reason") == "swap_limit" and "_resource_policy" not in config:
        raise ValueError("swap recovery requires reviewed resource policy")
    if "_resource_policy" in config:
        admission = _resource_admission(run, config, current)
        atomic(
            run / "last-resume.json",
            encoded(
                {
                    "at": time.time(),
                    "status": "admitted" if admission["safe"] else "blocked",
                    "reason": admission["reason"],
                    "previous_attempt": current.get("execution_attempt"),
                    "previous_stop_reason": current.get("reason"),
                    "operation_revision": ref,
                    "resource_admission": admission,
                }
            ),
        )
        if not admission["safe"]:
            # Preserve a fatal record verbatim; rejected admission never clears it.
            if current["status"] != "needs_astra":
                current.update(
                    status="stopped",
                    reason="requested_stop"
                    if admission["reason"] == "requested_stop"
                    else "resource_wait",
                    resource_wait=admission,
                )
                _save_state(run, current)
            return config, {**current, "resume_blocked": admission}
    else:
        admission = None
    attempts = run / "attempts"
    number = 1 + max(
        (int(p.stem) for p in attempts.glob("[0-9]*.json") if p.stem.isdigit()), default=0
    )
    record = {
        "schema": "open_shogiai_execution_attempt/v1",
        "number": number,
        "at": time.time(),
        "previous_state": current,
        "snapshot": snapshot,
        "operation_revision": _reference(revision_path) if revision_path else None,
        "interpreter": sys.executable,
        "cwd": str(ROOT),
        "stdin": "DEVNULL",
        "logs": {},
    }
    if admission is not None:
        record["resource_baseline"] = {**admission["samples"][-1], "attempt": number}
        record["resource_admission"] = admission
    for name in ("supervisor.log", f"generate-{current.get('retries', {}).get('generate', 0)}.log"):
        path = run / name
        if path.exists():
            record["logs"][name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    atomic(attempts / f"{number:06d}.json", encoded(record))
    pending = {
        k: v
        for k, v in current.items()
        if k in ("schema", "run_id", "run_sha256", "began_at", "initial_swap_bytes", "retries")
    }
    pending.update(
        status="running",
        reason="startup_pending",
        execution_attempt=number,
        **({"operation_revision": record["operation_revision"]} if revision_path else {}),
    )
    if admission is not None:
        pending["resource_baseline"] = record["resource_baseline"]
        pending["resource_history"] = current.get("resource_history", {})
    _save_state(run, pending)
    return config, pending


def _process_table() -> dict[int, tuple[int, int, str, int]]:
    result = subprocess.run(
        ["/bin/ps", "-axo", "pid=,ppid=,rss=,stat=,pgid="],
        capture_output=True,
        text=True,
        timeout=3,
        check=True,
        env={**os.environ, "LC_ALL": "C"},
    )
    if len(result.stdout) > 8 * 1024 * 1024:
        raise ValueError("process inventory exceeds bound")
    rows = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 5:
            raise ValueError("invalid process inventory")
        pid, parent, rss, group = (*(int(value) for value in fields[:3]), int(fields[4]))
        if min(pid, parent, rss, group) < 0 or pid == 0 or pid in rows:
            raise ValueError("invalid process inventory values")
        rows[pid] = parent, rss * 1024, fields[3], group
    return rows


def _stage_group_snapshot(process, expected_identity: str | None = None) -> dict:
    """Authenticate the still-unreaped direct child before inspecting its group.

    Do not call Popen.poll/send_signal/wait here: poll reaps an exited leader.
    Holding its zombie reserves the numeric PID/PGID until all group members have
    stopped. Darwin no longer exposes a zombie's start identity; its direct PPID
    and private PGID plus our unreaped Popen ownership are the remaining anchor.
    """
    if process.returncode is not None:
        raise RuntimeError("stage leader was reaped before group cleanup")
    rows = _process_table()
    leader = rows.get(process.pid)
    if leader is None or leader[0] != os.getpid() or leader[3] != process.pid:
        raise RuntimeError("cannot authenticate the unreaped stage process group")
    if not leader[2].startswith("Z"):
        identity = read_process_identity(process.pid)
        if identity is None:
            # The leader can exit between ps and the identity read. Reinspect it
            # without reaping; any other unexplained identity loss fails closed.
            rows = _process_table()
            leader = rows.get(process.pid)
            if (
                leader is None
                or leader[0] != os.getpid()
                or leader[3] != process.pid
                or not leader[2].startswith("Z")
            ):
                raise RuntimeError("active stage process identity unavailable")
        elif expected_identity is not None and identity != expected_identity:
            raise RuntimeError("stage process identity changed")
    return rows


def _sample_owned(process, owned: dict[int, str]) -> tuple[int, dict[int, str]]:
    rows = _stage_group_snapshot(process, owned.get(process.pid))
    identities = {}
    for pid, (_, _, state, group) in rows.items():
        if group != process.pid or state.startswith("Z"):
            continue
        identity = read_process_identity(pid)
        if identity is not None:
            # read_process_identity binds its current group as well as start time.
            if identity.split(":")[2] != str(process.pid):
                raise RuntimeError("stage member left the isolated process group")
            identities[pid] = identity
    # Charge every live member from the authenticated group inventory, including
    # short-lived helpers that exit before their individual identity can be read.
    return sum(
        row[1] for row in rows.values() if row[3] == process.pid and not row[2].startswith("Z")
    ), identities


def _signal_owned(owned: dict[int, str], requested_signal: signal.Signals) -> list[int]:
    signalled = []
    for pid, identity in owned.items():
        if pid <= 1 or pid == os.getpid() or read_process_identity(pid) != identity:
            continue
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, requested_signal)
            signalled.append(pid)
    return signalled


def _cleanup(process, owned: dict[int, str], *, grace_seconds: float = 20) -> dict:
    expected_identity = owned.get(process.pid)
    signalled = set()

    def active_members():
        rows = _stage_group_snapshot(process, expected_identity)
        return {
            pid: row
            for pid, row in rows.items()
            if row[3] == process.pid and not row[2].startswith("Z")
        }

    # Let STOP-aware stages flush a checkpoint without reaping their leader.
    deadline = time.monotonic() + grace_seconds
    active = active_members()
    while active and time.monotonic() < deadline:
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        active = active_members()
    for requested_signal in (signal.SIGTERM, signal.SIGKILL):
        if not active:
            break
        # The leader is still our unreaped child, so this PGID cannot be reused.
        # killpg includes children created after the inventory and before the signal.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, requested_signal)
            signalled.update(active)
        deadline = time.monotonic() + 3
        while active and time.monotonic() < deadline:
            time.sleep(0.05)
            active = active_members()
    if active:
        raise RuntimeError("stage group did not stop after bounded cleanup")
    # No live group member can fork now. Reap only after this verified empty state.
    process.wait(timeout=1)
    return {
        "signalled_pids": sorted(signalled),
        "remaining_processes": {},
        "inventory_errors": [],
        "group_cleanup_before_reap": True,
    }


def _register_stage_group(run: Path, stage: str) -> dict:
    if _residual_stage_group(run) is not None:
        raise RuntimeError("unproven residual stage group prevents child launch")
    pid, group, session = os.getpid(), os.getpgid(0), os.getsid(0)
    identity = read_process_identity(pid)
    if pid != group or pid != session or identity is None:
        raise RuntimeError("stage must start in its own authenticated process group/session")
    receipt = {
        "schema": "open_shogiai_evaluator_stage_process/v1",
        "run_sha256": digest(run / "run.json"),
        "stage": stage,
        "pid": pid,
        "pgid": group,
        "process_identity": identity,
        "registered_at": time.time(),
        "status": "active",
    }
    # This must precede generation, native probes, torch or any other child launch.
    atomic(run / "stage-process.json", encoded(receipt))
    return receipt


def _residual_stage_group(run: Path) -> dict | None:
    path = run / "stage-process.json"
    if not path.exists():
        return None
    receipt = _json(path)
    pid = receipt.get("pid")
    if (
        receipt.get("schema") != "open_shogiai_evaluator_stage_process/v1"
        or receipt.get("run_sha256") != digest(run / "run.json")
        or type(pid) is not int
        or pid <= 1
        or receipt.get("pgid") != pid
        or not isinstance(receipt.get("process_identity"), str)
        or receipt.get("status") not in ("active", "cleaned")
    ):
        raise ValueError("invalid persistent stage group receipt")
    if receipt["status"] == "cleaned":
        return None
    members = [process_id for process_id, row in _process_table().items() if row[3] == pid]
    if members:
        # Without the original supervisor's unreaped Popen, do not authorize a
        # numeric-group signal. Preserve evidence and forbid another writer.
        return {**receipt, "remaining_pids": sorted(members)}
    return None


def _mark_group_cleaned(run: Path, process) -> None:
    path = run / "stage-process.json"
    if not path.exists():
        return  # The stage exited before registering, hence before creating children.
    receipt = _json(path)
    if receipt["pid"] != process.pid:
        if receipt["status"] != "cleaned":
            raise RuntimeError("stage group receipt belongs to another active process")
        return
    receipt.update(status="cleaned", cleaned_at=time.time(), returncode=process.returncode)
    atomic(path, encoded(receipt))


def status(run: Path) -> dict:
    value = _state(run) if (run / "state.json").exists() else {}
    value["process_alive"] = bool(value.get("process_identity")) and (
        read_process_identity(value.get("pid")) == value["process_identity"]
    )
    value["stage_alive"] = bool(value.get("stage_identity")) and (
        read_process_identity(value.get("stage_pid")) == value["stage_identity"]
    )
    group = _residual_stage_group(run)
    if group is not None:
        value["stage_group"] = group
        value["stage_alive"] = read_process_identity(group["pid"]) == group["process_identity"]
    try:
        with _lease(run):
            value["lease_held"] = False
    except BlockingIOError:
        value["lease_held"] = True
    if value["lease_held"] and not value["process_alive"]:
        value["supervision"] = "supervisor_missing; retained stage lease prevents another writer"
    if (run / "last-resume.json").exists():
        value["latest_resume"] = _json(run / "last-resume.json")
    value["completed_trajectories"] = len(list((run / "data" / "games").glob("*.receipt.json")))
    progress_path = run / "data" / "generation-progress.json"
    if progress_path.exists():
        value["generation"] = _json(progress_path)
        value["generation"]["games"] = value["completed_trajectories"]
    valid_files = list((run / "data" / "games").glob("*.receipt.json"))
    value["last_valid_output_at"] = max((p.stat().st_mtime for p in valid_files), default=None)
    if (run / "fit" / "progress.json").exists():
        value["training"] = _json(run / "fit" / "progress.json")
    if (run / "data" / "dataset" / "manifest.json").exists():
        value["unique_positions"] = _json(run / "data" / "dataset" / "manifest.json")[
            "unique_positions"
        ]
    if (run / "fit" / "training.json").exists():
        summary = _json(run / "fit" / "training.json")
        value.update(training_status=summary["status"], best_sha256=summary["best_sha256"])
    return value


def diagnose(run: Path, *, _lease_held: bool = False) -> dict:
    """Read-only resume checks; distinguish previous failure from present admission."""
    report = {"at": time.time(), "checks": {}, "unchecked": [], "resume": "blocked"}
    checks = report["checks"]

    def check(name, action):
        try:
            value = action()
            checks[name] = {"status": "pass"}
            return value
        except (
            OSError,
            ValueError,
            RuntimeError,
            KeyError,
            sqlite3.Error,
            subprocess.SubprocessError,
        ) as error:
            checks[name] = {"status": "blocked", "reason": f"{type(error).__name__}:{error}"}
            return None

    current = check("state", lambda: _state(run))
    if current:
        report["previous_attempt"] = {
            "number": current.get("execution_attempt"),
            "status": current["status"],
            "reason": current.get("reason"),
        }
    if (run / "last-resume.json").exists():
        report["latest_resume"] = check(
            "last_resume_record", lambda: _json(run / "last-resume.json")
        )
    try:
        with contextlib.nullcontext() if _lease_held else _lease(run):
            checks["lease"] = {"status": "pass"}
            check("idle_processes_and_recovery", lambda: _resume_idle_state(run))

            def approved():
                ref = _json(run / "approved-operation.json")
                path = inside(ref["path"])
                if path.parent != run / "operations":
                    raise ValueError("approved operation outside run")
                _reference(path, ref["sha256"])
                _reference(run / "run.json", _json(run / "seal.json")["run_sha256"])
                config = _json(run / "run.json")
                _validate_config(config)
                revision = _json(path)
                _operation_code(run, config, revision)
                check("approved_code_seal_inputs", lambda: verify(run, operation_revision=revision))

                def committed_code():
                    if _code_identity()["files"] != {
                        **config["code"]["files"],
                        **revision["files"],
                    }:
                        raise ValueError(
                            "resume code differs from Astra-approved operation revision"
                        )

                check("committed_code", committed_code)
                return config

            config = check("approved_operation", approved)
            if config is not None:
                check("migration", lambda: _require_migration(run, config))
                snapshot = check(
                    "data_receipts_ledger_cursor", lambda: _resume_snapshot(run, config)
                )
                if snapshot:
                    report["progress"] = {k: v for k, v in snapshot.items() if k != "files"}
                for stage in _stages(config):
                    if (run / f"{stage}-complete.json").exists():
                        check(
                            stage + "_completion",
                            lambda stage=stage: _verify_completion(run, stage, config),
                        )
                if current:
                    checks["calendar"] = {
                        "status": "blocked" if _calendar_expired(config, current) else "pass"
                    }
                    if checks["calendar"]["status"] == "blocked":
                        checks["calendar"]["reason"] = "wall_limit"
                    report["time_accounting"] = {
                        "calendar_policy": config.get("_calendar_policy"),
                        "historical_began_at": current.get("began_at"),
                        "historical_calendar_limit_seconds": config["resources"][
                            "maximum_wall_seconds"
                        ],
                        "calendar_elapsed_seconds": time.time()
                        - current.get("began_at", time.time()),
                        "total_active_seconds": None,
                    }
            else:
                report["unchecked"].extend(["migration", "data", "stage_completions", "calendar"])
            observed = check(
                "resource_measurement",
                lambda: _resource_sample(run, _process_table()[os.getpid()][1]),
            )
            if observed:
                report["current_resource_sample"] = observed
                if config is not None:
                    condition = _resource_condition(observed, None, config, admission=True)
                    checks["current_resource_condition"] = {
                        "status": "pass" if condition == "warming" else "blocked",
                        "reason": condition,
                    }
            report["unchecked"].append("fresh consecutive resource admission at resume")
    except BlockingIOError:
        checks["lease"] = {"status": "blocked", "reason": "another writer holds lease"}
        report["unchecked"].extend(["code", "data", "resources"])
    if all(v["status"] == "pass" for v in checks.values()):
        report["resume"] = "eligible_pending_resource_admission"
    return report


def _clear_stop(run: Path) -> None:
    for path in (run / "STOP", run / "data" / "STOP", run / "arena" / "STOP"):
        if path.is_symlink() or (path.exists() and (not path.is_file() or path.stat().st_size)):
            raise ValueError("unexpected STOP marker")
        path.unlink(missing_ok=True)


def _require_migration(run: Path, config: dict) -> None:
    if config.get("recovery_from") and config["generation"].get("recovery_policy"):
        receipt = _json(run / "migration.json")
        if receipt.get("run_sha256") != digest(run / "run.json"):
            raise ValueError("migration belongs to another run")
        _reference(run / "data/inherited.json", receipt["inherited_sha256"])


def migrate(run: Path) -> dict:
    """One idempotent parent-shard and failure-history import before any execution."""
    from .defense_scenarios import assignment, rotate_sfen
    from .evaluator_data import START
    from .evaluator_ledger import export_queue, import_failure, migrate_generation

    with _lease(run):
        config = verify(run)
        if (run / "migration.json").exists():
            _require_migration(run, config)
            return _json(run / "migration.json")
        if _state(run)["status"] != "ready_for_luna" or _residual_stage_group(run) is not None:
            raise ValueError("migration requires a new idle successor")
        parent = config["recovery_from"]
        parent_run = inside(parent["path"])
        with _lease(parent_run):
            if status(parent_run)["process_alive"] or _residual_stage_group(parent_run) is not None:
                raise ValueError("parent writer is active")
            generation = config["generation"]
            report = migrate_generation(ROOT, parent_run / "data", run / "data", generation)
            failures = []
            for ref in parent["failures"]:
                path = inside(ref["path"])
                _reference(path, ref["sha256"])
                evidence = _json(path)
                if (
                    evidence["game"],
                    evidence["ply"],
                    evidence["branch"],
                    evidence["error_type"],
                ) != (211, 116, "root", "USIIncompleteDepthError"):
                    raise ValueError("failure is outside the reviewed recovery task")
                failures.append(path)
            if len(failures) != 2:
                raise ValueError("both consumed teacher attempts are required")
            _, variant = assignment(generation, 211)
            imported = import_failure(
                run / "data", generation, failures, rotate_sfen(START) if variant % 2 else START
            )
            atomic(run / "data/generation.json", encoded(generation))
            export_queue(run / "data")
            receipt = {
                "schema": "open_shogiai_recovery_migration/v1",
                "run_sha256": digest(run / "run.json"),
                "inherited_sha256": digest(run / "data/inherited.json"),
                "inherited_games": len(report["games"]),
                "imported_failure": imported,
            }
            atomic(run / "migration.json", encoded(receipt))
            return receipt


def start(
    run: Path,
    *,
    pause_after_new_games: int | None = None,
    recover_startup: bool = False,
    pause_after_arena_games: int | None = None,
    pause_after_updates: int | None = None,
) -> dict:
    if pause_after_new_games is not None:
        _number(pause_after_new_games, 1, 12, "probe trajectory bound")
    if pause_after_arena_games is not None:
        _number(pause_after_arena_games, 2, 36, "arena completed-task pause bound")
        if pause_after_arena_games % 2 or pause_after_new_games is not None:
            raise ValueError("arena probe requires a complete color pair and no generation probe")
    if pause_after_updates is not None:
        _number(pause_after_updates, 1, 32, "initial training update prefix")
        if pause_after_new_games is not None or pause_after_arena_games is not None:
            raise ValueError("one bounded prefix per invocation")
    try:
        with _lease(run) as lease:
            if recover_startup:
                try:
                    config, current = _prepare_resume(run)
                except (
                    OSError,
                    ValueError,
                    RuntimeError,
                    KeyError,
                    sqlite3.Error,
                    subprocess.SubprocessError,
                ) as error:
                    report = diagnose(run, _lease_held=True)
                    _record_resume(
                        run,
                        "blocked",
                        str(error),
                        checks=report["checks"],
                        unchecked=report["unchecked"],
                    )
                    raise
                if current.get("resume_blocked"):
                    return current
            else:
                config = verify(run)
                _require_migration(run, config)
                current = _state(run)
                residual = _residual_stage_group(run)
                if residual is not None:
                    current.update(
                        status="needs_astra",
                        reason="residual_stage_group_unproven",
                        residual_group=residual,
                    )
                    _save_state(run, current)
                    return current
                if current["status"] in {
                    "needs_astra",
                    "awaiting_astra_browser",
                    "awaiting_astra_review",
                }:
                    return current
            try:
                _clear_stop(run)
                process = _launch(
                    run,
                    [
                        sys.executable,
                        "-m",
                        MODULE,
                        "work",
                        str(run.relative_to(ROOT)),
                        "--lease-fd",
                        str(lease),
                        *(
                            ["--pause-after-new-games", str(pause_after_new_games)]
                            if pause_after_new_games is not None
                            else []
                        ),
                        *(
                            ["--pause-after-updates", str(pause_after_updates)]
                            if pause_after_updates is not None
                            else []
                        ),
                        *(
                            ["--pause-after-arena-games", str(pause_after_arena_games)]
                            if pause_after_arena_games is not None
                            else []
                        ),
                    ],
                    "supervisor.log",
                    lease,
                )
            except (OSError, ValueError) as error:
                _remember_error(current, "supervisor_spawn", error)
                current.update(
                    status="needs_astra",
                    reason="supervisor_spawn_failed",
                    startup_failure="before_spawn",
                )
                _save_state(run, current)
                if recover_startup:
                    _record_resume(
                        run,
                        "blocked",
                        current["reason"],
                        execution_attempt=current.get("execution_attempt"),
                    )
                raise
            if recover_startup:
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    observed = _state(run)
                    if observed.get("pid") == process.pid:
                        _record_resume(
                            run,
                            "started",
                            "supervisor_acknowledged",
                            execution_attempt=observed.get("execution_attempt"),
                            pid=process.pid,
                        )
                        return observed
                    if process.poll() is not None:
                        current.update(
                            status="needs_astra",
                            reason="supervisor_bootstrap_failed",
                            supervisor_returncode=process.returncode,
                        )
                        _save_state(run, current)
                        _record_resume(
                            run,
                            "blocked",
                            current["reason"],
                            execution_attempt=current.get("execution_attempt"),
                        )
                        return current
                    time.sleep(0.05)
                # Retain the live child's lease and pending attempt; never spawn again.
                _record_resume(
                    run,
                    "unconfirmed",
                    "supervisor_acknowledgement_timeout",
                    execution_attempt=current.get("execution_attempt"),
                    pid=process.pid,
                )
                raise RuntimeError("supervisor acknowledgement timed out; inspect status and pause")
            return {
                "status": "starting",
                "pid": process.pid,
                "run_id": config["run_id"],
                "log": str((run / "supervisor.log").relative_to(ROOT)),
            }
    except KeyboardInterrupt:
        stop(run)
        raise
    except BlockingIOError:
        current = status(run)
        if recover_startup:
            current["resume_blocked"] = {"safe": False, "reason": "lease_held"}
        if not current["process_alive"] and (current["stage_alive"] or current.get("stage_group")):
            stop(run)
            current["status"] = "supervisor_missing_stop_requested"
        return current


def pause(run: Path) -> dict:
    stop(run)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            with _lease(run):
                current = _state(run)
                if any(
                    current.get(key) and read_process_identity(current.get(pid)) == current[key]
                    for pid, key in (("pid", "process_identity"), ("stage_pid", "stage_identity"))
                ):
                    time.sleep(0.05)
                    continue
                if _residual_stage_group(run) is not None:
                    raise ValueError("pause has an unproven residual group")
                if current["status"] not in {"stopped", "ready_for_luna"}:
                    raise ValueError("pause cannot clear a failure or unfinished startup")
                try:
                    from .evaluator_ledger import export_queue

                    config = verify(run)
                    snapshot = _resume_snapshot(run, config)
                    if not (run / "generate-complete.json").exists():
                        export_queue(run / "data")
                    progress = {k: snapshot[k] for k in ("games", "rows", "tasks")}
                    atomic(
                        run / "data/generation-progress.json",
                        encoded({**progress, "status": "paused"}),
                    )
                    current.update(
                        status="ready_for_luna", reason="requested_pause", generation=progress
                    )
                    if current.get("execution_attempt"):
                        atomic(
                            run / "attempts" / f"{current['execution_attempt']:06d}-pause.json",
                            encoded({"state": current, "snapshot": snapshot}),
                        )
                except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
                    _remember_error(current, "pause", error)
                    current.update(status="needs_astra", reason="pause_verification_failed")
                    _save_state(run, current)
                    raise
                _save_state(run, current)
                return current
        except BlockingIOError:
            time.sleep(0.1)
    raise RuntimeError("pause did not release the lease within its deadline")


def stop(run: Path) -> dict:
    for folder in (run, run / "data", run / "arena"):
        if not folder.exists():
            continue
        path = inside(folder / "STOP", exists=False)
        if path.exists() and (not path.is_file() or path.stat().st_size):
            raise ValueError("unexpected STOP marker")
        path.touch()
    return {
        "status": "stop_requested",
        "retention": "completed data, best model and latest resume states",
    }


def _apply_resource_epoch(inherited: dict, epoch: dict) -> dict:
    if (
        epoch.get("authorization") != "explicit_user_approval_20260912"
        or inherited["initial_swap_bytes"] != epoch["original_initial_swap_bytes"]
    ):
        raise ValueError("resource epoch does not match the approved parent baseline")
    return {
        **inherited,
        "original_initial_swap_bytes": inherited["initial_swap_bytes"],
        "initial_swap_bytes": epoch["initial_swap_bytes"],
        "resource_epoch": epoch,
    }


def _memory_free_percent() -> int:
    output = subprocess.check_output(["memory_pressure", "-Q"], text=True, timeout=3)
    found = re.search(r"System-wide memory free percentage:\s*(\d+)%", output)
    if found is None or not 0 <= int(found[1]) <= 100:
        raise ValueError("memory pressure measurement unavailable")
    return int(found[1])


def _swap_bytes() -> int:
    output = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True, timeout=3)
    found = re.search(r"used\s*=\s*([0-9.]+)([MG])", output)
    if found is None:
        raise ValueError("swap measurement unavailable")
    return int(float(found[1]) * (1024**2 if found[2] == "M" else 1024**3))


def _resource_sample(run: Path, rss: int) -> dict:
    """Read-only macOS gauges and boot-local counters; RSS is an upper approximation."""
    started = time.monotonic()
    vm = subprocess.check_output(["vm_stat"], text=True, timeout=3)
    page = re.search(r"page size of (\d+) bytes", vm)
    outs = re.search(r"^Swapouts:\s*(\d+)\.", vm, re.MULTILINE)
    boot = subprocess.check_output(["sysctl", "-n", "kern.boottime"], text=True, timeout=3).strip()
    pressure = int(
        subprocess.check_output(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True, timeout=3
        )
    )
    if not page or not outs or not boot or pressure not in (1, 2, 4):
        raise ValueError("resource measurement unavailable")
    sample = {
        "at": time.time(),
        "monotonic": started,
        "boot": boot,
        "page_size": int(page[1]),
        "swapouts": int(outs[1]),
        "pressure": pressure,
        "swap_bytes": _swap_bytes(),
        "memory_free_percent": _memory_free_percent(),
        "free_bytes": shutil.disk_usage(run).free,
        "rss_bytes": rss,
        "source": "sysctl vm.swapusage used (rounded MiB); vm_stat Swapouts pages; "
        "sysctl kern.boottime and pressure dispatch flags; memory_pressure -Q; ps RSS KiB",
        "units": "bytes; swapouts=pages; pressure=flags; memory_free_percent=percent; time=seconds",
    }
    sample["at"] = time.time()
    if time.monotonic() - started > RESOURCE_POLICY["maximum_sample_age_seconds"]:
        raise ValueError("stale resource sample")
    return sample


def _resource_condition(
    sample: dict, previous: dict | None, config: dict, *, admission=False
) -> str:
    policy, limits = RESOURCE_POLICY, config["resources"]
    if "_dataset_admission" in config or "round" in config:
        # Host pressure flags (1=normal,2=warning,4=critical) and swap are diagnostics.
        return (
            "critical:disk"
            if sample["free_bytes"] < limits["free_space_floor_gib"] * 1024**3
            else "safe"
        )
    if not 0 <= time.time() - sample["at"] <= policy["maximum_sample_age_seconds"]:
        return "critical:stale_sample"
    if sample["pressure"] not in (1, 2, 4):
        return "critical:unknown_pressure"
    if (
        sample["pressure"] == 4
        or sample["memory_free_percent"] < policy["critical_memory_free_percent"]
    ):
        return "critical:memory_pressure"
    if sample["free_bytes"] < limits["free_space_floor_gib"] * 1024**3:
        return "critical:disk"
    if sample["rss_bytes"] > limits["maximum_process_rss_gib"] * 1024**3:
        return "critical:process_rss"
    if sample["pressure"] != 1 or sample["memory_free_percent"] < limits.get(
        "minimum_memory_free_percent", 50
    ):
        return "memory_pressure"
    if previous is None:
        return "warming"
    elapsed = sample["monotonic"] - previous["monotonic"]
    if (
        sample["boot"] != previous["boot"]
        or sample["page_size"] != previous["page_size"]
        or sample["swapouts"] < previous["swapouts"]
        or not 0 < elapsed <= 2 * policy["maximum_sample_age_seconds"]
    ):
        return "critical:counter_discontinuity"
    # These are host-wide rates, never attributed to the owned process tree.
    rate = (sample["swapouts"] - previous["swapouts"]) * sample["page_size"] / elapsed
    growth = (sample["swap_bytes"] - previous["swap_bytes"]) / elapsed
    threshold = (
        policy["quiet_bytes_per_second"] if admission else policy["swapout_bytes_per_second"]
    )
    if rate > threshold or growth > threshold:
        return "swap_activity"
    if admission and (
        sample["memory_free_percent"] < policy["resume_memory_free_percent"]
        or sample["free_bytes"]
        < (limits["free_space_floor_gib"] + policy["resume_disk_margin_gib"]) * 1024**3
        or sample["rss_bytes"]
        > limits["maximum_process_rss_gib"] * 1024**3 * policy["resume_rss_fraction"]
    ):
        return "recovery_margin"
    return "safe"


def _calendar_expired(config: dict, state: dict) -> bool:
    # Calendar age is historical metadata, not active computation consumption.
    return (
        "round" not in config
        and "_calendar_policy" not in config
        and (
            time.time()
            >= state.get("began_at", time.time()) + config["resources"]["maximum_wall_seconds"]
        )
    )


def _resource_admission(run: Path, config: dict, state: dict) -> dict:
    """Finite pre-launch wait; no teacher/task wall budget runs during admission."""
    started, samples, consecutive, reason = time.monotonic(), [], 0, "warming"
    stop_path = run / "STOP"
    stop_stamp = stop_path.stat().st_mtime_ns if stop_path.exists() else None
    for index in range(RESOURCE_POLICY["maximum_wait_samples"]):
        if (stop_path.stat().st_mtime_ns if stop_path.exists() else None) != stop_stamp:
            reason = "requested_stop"
            break
        if _calendar_expired(config, state):
            reason = "wall_limit"
            break
        try:
            # Admission holds the exclusive lease and has checked absence of all old children.
            table = _process_table()
            sample = _resource_sample(run, table[os.getpid()][1])
            reason = _resource_condition(
                sample, samples[-1] if samples else None, config, admission=True
            )
            samples.append(sample)
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            reason = f"measurement_unavailable:{error}"
            consecutive = 0
            break
        consecutive = consecutive + 1 if reason == "safe" else 0
        if (stop_path.stat().st_mtime_ns if stop_path.exists() else None) != stop_stamp:
            reason, consecutive = "requested_stop", 0
            break
        if consecutive >= RESOURCE_POLICY["admission_samples"]:
            break
        if reason.startswith("critical:"):
            break
        if index + 1 < RESOURCE_POLICY["maximum_wait_samples"]:
            time.sleep(RESOURCE_POLICY["interval_seconds"])
    result = {
        "safe": consecutive >= RESOURCE_POLICY["admission_samples"],
        "reason": reason,
        "samples": samples,
        "wait_seconds": time.monotonic() - started,
        "at": time.time(),
        "policy": RESOURCE_POLICY,
        "previous_attempt": state.get("execution_attempt"),
    }
    with inside(run / "resource-admission.jsonl", exists=False).open("ab") as stream:
        stream.write(encoded(result) + b"\n")
    return result


def _progress_signature(run: Path, stage: str) -> tuple:
    folder = {
        "generate": run / "data" / "games",
        "prepare": run / "data" / "dataset",
        "train": run / "fit",
        "audit": run,
        "arena": run / "arena",
        "integrate": run / "development",
    }[stage]
    ignored = {
        "state.json",
        "stage-process.json",
        "supervisor.log",
        "monitor.jsonl",
        "operation.lock",
    }
    observed = []
    for path in folder.rglob("*") if folder.exists() else []:
        if path.name in ignored or path.name.endswith((".log", ".tmp")):
            continue
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue  # Atomic publication and bounded checkpoint pruning are normal.
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("unexpected symlink in stage output")
        if stat.S_ISREG(info.st_mode):
            observed.append((info.st_size, info.st_mtime_ns))
    if stage == "generate" and (run / "data/tasks.sqlite3").exists():
        with sqlite3.connect(
            f"file:{inside(run / 'data/tasks.sqlite3')}?mode=ro", uri=True, timeout=3
        ) as db:
            exists = db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='tasks'"
            ).fetchone()
            count, updated = (
                db.execute(
                    "SELECT count(*),max(updated) FROM tasks WHERE status='accepted'"
                ).fetchone()
                if exists
                else (0, None)
            )
        # Only accepted task publication is progress; retry/heartbeat writes are not.
        if count:
            observed.append((count, int(updated * 1_000_000_000)))
    return (
        len(observed),
        sum(size for size, _ in observed),
        max((modified for _, modified in observed), default=0),
    )


def _dataset(run: Path, config: dict) -> dict:
    if config["generation"].get("prepared_dataset"):
        from .r4_data import verify_dataset

        return verify_dataset(
            ROOT, run / "data/dataset", config["generation"]["prepared_dataset"]["manifest_sha256"]
        )
    if not (run / "data" / "dataset" / "manifest.json").exists():
        raise ValueError("completed dataset manifest is required")
    manifest = _json(run / "data" / "dataset" / "manifest.json")
    for name, directory in (
        ("artifacts", run / "data" / "dataset"),
        ("source_games", run / "data" / "games"),
    ):
        for ref in manifest[name]:
            path = Path(ref["path"])
            if path.is_absolute() or len(path.parts) != 1:
                raise ValueError("dataset references must be local file names")
            inside(directory / path)  # Reject links before prepare reads the referenced bytes.
    result = prepare(ROOT, run / "data", config["generation"], config["excluded_development_sfens"])
    if config["generation"].get("recovery_policy"):
        from .evaluator_coverage import coverage_report

        coverage = coverage_report(run / "data", config["generation"], result)
        if not coverage["passed"]:
            raise ValueError(
                "finite recovery coverage insufficient: " + ",".join(coverage["reasons"])
            )
    if config["generation"].get("defense_campaign"):
        goal = config["unique_data_goal"]
        if (
            result["unique_positions"]["train"] < goal["minimum_train_positions"]
            or result["generated_unique_positions"]["train"] < goal["minimum_new_train_positions"]
        ):
            raise ValueError("insufficient unique training data; return to Astra without fitting")
        import numpy as np

        for split in ("validation", "development_test"):
            groups = np.load(run / "data/dataset" / f"{split}-groups.npy", allow_pickle=False)
            if any(
                int((groups == i).sum()) < goal["minimum_evaluation_rows_per_group"]
                for i in range(4)
            ):
                raise ValueError("insufficient unknown evaluation coverage")
        train_groups = np.load(run / "data/dataset/train-groups.npy", allow_pickle=False)
        if any(
            int((train_groups == i).sum()) < goal["minimum_train_rows_per_group"] for i in range(4)
        ):
            raise ValueError("insufficient training stratum; refuse small-set exposure inflation")
    return result


def _prepare_recovery(run: Path, config: dict) -> dict:
    """Freeze each attempted manifest; supplement once within the sealed range."""
    from .evaluator_coverage import coverage_report

    data = run / "data"
    generation = config["generation"]
    transaction = data / "manifest-supplement.json"
    if transaction.exists():
        pending = _json(transaction)
        _hash(pending["manifest_sha256"])
        archive = inside(data / "manifest-attempts" / pending["manifest_sha256"])
    else:
        manifest = prepare(ROOT, data, generation, config["excluded_development_sfens"])
        coverage = coverage_report(data, generation, manifest)
        if coverage["passed"]:
            return _dataset(run, config)
        manifest_sha = digest(data / "dataset/manifest.json")
        archive = data / "manifest-attempts" / manifest_sha
        archive.mkdir(parents=True, exist_ok=True)
        pending = {"manifest_sha256": manifest_sha, "phase": "archiving", "coverage": coverage}
        atomic(transaction, encoded(pending))
    if pending["phase"] == "archiving":
        # Persist the journal before either rename. Existing destinations mean that
        # individual rename committed before interruption; never re-create that source.
        for name in ("dataset-generated", "dataset"):
            source, destination = data / name, archive / name
            if destination.exists():
                if source.exists():
                    raise ValueError("manifest archive has conflicting source and destination")
                continue
            if source.exists():
                source.rename(destination)
        _reference(archive / "dataset/manifest.json", pending["manifest_sha256"])
        atomic(archive / "coverage.json", encoded(pending["coverage"]))
        pending["phase"] = "supplementing"
        atomic(transaction, encoded(pending))
    if pending["phase"] != "supplementing":
        raise ValueError("unknown manifest supplement transaction")
    result = generate(ROOT, data, generation, supplemental=True)
    if result.get("status") != "complete":
        raise ValueError("finite supplemental generation exhausted: " + str(result.get("status")))
    prepare(ROOT, data, generation, config["excluded_development_sfens"])
    result = _dataset(run, config)
    transaction.unlink()
    return result


def _training_identity(run: Path, config: dict) -> dict:
    return {
        "code_commit": config["code"]["commit"],
        **(
            {
                "dataset_admission": config["_dataset_admission"],
                "operation_revision": config.get("_post_training", {}).get(
                    "training_operation", _state(run)["operation_revision"]
                ),
            }
            if "_dataset_admission" in config
            else {}
        ),
        "run_sha256": digest(run / "run.json"),
        "dataset_sha256": digest(run / "data/dataset/manifest.json"),
    }


def selected_model(run: Path) -> Path:
    config = _json(run / "run.json")
    name = (
        "trained_candidate.osaval03"
        if config.get("training", {}).get("trained_candidate")
        else "best.osaval03"
    )
    return run / "fit" / name


def _training_summary(run: Path, config: dict) -> dict:
    result = _json(run / "fit" / "training.json")
    expected = _training_identity(run, config)
    if result["status"] != "complete" or result["identity"] != expected:
        raise ValueError("training completion identity mismatch")
    _reference(run / "fit/best.osaval03", result["best_sha256"])
    if config["training"].get("trained_candidate"):
        candidate = result.get("trained_candidate")
        if (
            not candidate
            or candidate["step"] <= 0
            or candidate["sha256"] == config["generation"]["leaf_sha256"]
        ):
            raise ValueError("C3 requires a distinct optimizer-updated candidate")
        _reference(selected_model(run), candidate["sha256"])
    resume = _json(run / "fit" / "resume.json")
    if resume["step"] != result["step"]:
        raise ValueError("final checkpoint step differs from training completion")
    checkpoint = inside(run / "fit" / resume["path"])
    if checkpoint.parent != run / "fit":
        raise ValueError("checkpoint escapes fit directory")
    _reference(checkpoint, resume["sha256"])
    return result


def _audit_report(
    run: Path, config: dict, *, report: Path | None = None, model: Path | None = None
) -> dict:
    value = _json(report or run / "model-audit.json")
    if (
        value.get("schema") != "open_shogiai_evaluator_export_audit/v1"
        or value.get("runtime") != "actual-wasm-in-node"
        or value.get("parity", {}).get("roots", 0) < 10
        or value.get("parity", {}).get("children", 0) < 1
        or value.get("parity", {}).get("maximumCpDifference") != 0
        or len(value.get("searches", [])) != 2
        or value.get("status") != "PASS"
        or value.get("format") != "OSAVAL03"
        or value.get("profile") != "pure_learned"
        or value.get("errors") != []
        or value.get("parity", {}).get("errors") != 0
    ):
        raise ValueError("existing model audit is not a successful export contract")
    paths = {
        "auditScript": ROOT / "scripts/check_evaluator_model.mjs",
        "moduleJs": inside(config["runtime"]["module"]["path"]),
        "wasm": inside(config["runtime"]["wasm"]["path"]),
        "nativeProbe": inside(config["runtime"]["replay"]["path"]),
        "model": model or selected_model(run),
    }
    for name, path in paths.items():
        _reference(path, value["artifacts"][name]["sha256"])
    return value


def _arena_complete(result: dict, expected_games: int = 40) -> None:
    if result.get("status") == "stopped":
        raise InterruptedError("arena stopped with retained game receipts")
    games = result.get("games", [])
    exhausted = (
        len(games) == expected_games
        and len({game["id"] for game in games}) == expected_games
        and all(
            game["status"] == "completed"
            or (game["status"] == "incomplete" and game.get("reason") == "max_plies_unscored")
            for game in games
        )
        and result.get("adoption_criteria_met") is False
    )
    if (
        result.get("status") != "complete"
        or result.get("planned_games") != expected_games
        or (result.get("summary", {}).get("all_planned_complete") is not True and not exhausted)
    ):
        raise ValueError("arena is failed or incomplete; preserve its evidence for Astra")
    # Finishing the finite workload is distinct from completing scored games.
    # Ply-exhausted games remain unscored and prevent strength acceptance, not review.


def _completion_artifacts(run: Path, stage: str) -> list[dict]:
    paths = {
        "generate": [
            run / "data" / "generation.json",
            run / "data" / "generation-complete.json",
            *sorted((run / "data" / "games").glob("*.json.gz")),
            *sorted((run / "data" / "games").glob("*.receipt.json")),
        ],
        "prepare": [run / "data" / "dataset" / "manifest.json"],
        "train": [
            run / "fit" / "training.json",
            run / "fit" / "resume.json",
            selected_model(run),
        ],
        "audit": [run / "model-audit.json", run / "development-test.json"],
        "arena": [run / "arena" / "arena.json", run / "arena" / "plan.json"],
        "integrate": [
            run / "development/result.json",
            run / "development/browser/browser.json",
            run / "development/model-audit.json",
        ],
    }[stage]
    if stage == "generate" and _json(run / "run.json")["generation"].get("recovery_policy"):
        paths.append(run / "data/recovery-queue.json")
    if stage == "arena":
        for attempt in _json(run / "arena" / "arena.json")["attempts"]:
            paths.extend(
                inside(ref["path"])
                for ref in [attempt["trace"], *attempt.get("stderr_artifacts", [])]
                if ref is not None
            )
    if stage == "train" and selected_model(run).name != "best.osaval03":
        paths.append(run / "fit/best.osaval03")
    if stage == "audit" and (run / "move-screen").exists():
        paths.extend(sorted((run / "move-screen").glob("*.json")))
    return [_reference(path) for path in paths]


def _verify_completion(run: Path, stage: str, config: dict) -> dict:
    receipt = _json(run / f"{stage}-complete.json")
    if (
        receipt.get("schema") != "open_shogiai_evaluator_stage/v1"
        or receipt.get("stage") != stage
        or receipt.get("run_sha256") != digest(run / "run.json")
    ):
        raise ValueError("completion receipt belongs to another run/stage")
    if receipt["artifacts"] != _completion_artifacts(run, stage):
        raise ValueError("completed stage artifact changed")
    if stage in ("prepare", "train", "audit", "arena", "integrate"):
        _dataset(run, config)
    if stage in ("train", "audit", "arena", "integrate"):
        _training_summary(run, config)
    if stage in ("audit", "arena", "integrate"):
        _audit_report(run, config)
    if stage == "arena":
        _arena_complete(
            receipt["result"], 32 + config["evaluation"]["startpos_demonstration_games"]
        )
        if receipt["result"] != _json(run / "arena" / "arena.json"):
            raise ValueError("arena summary differs from its completion receipt")
    if stage == "integrate":
        result = _json(run / "development/result.json")
        browser = _json(run / "development/browser/browser.json")
        model_sha = digest(selected_model(run))
        if (
            receipt["result"] != result
            or result.get("status") != "PASS"
            or result.get("rehearsal") is not False
            or result.get("run_sha256") != digest(run / "run.json")
            or result.get("model", {}).get("sha256") != model_sha
            or browser.get("status") != "PASS"
            or browser.get("expectedHash") != model_sha
            or result.get("browser_sha256") != digest(run / "development/browser/browser.json")
        ):
            raise ValueError("development completion is not the verified selected candidate")
    return receipt["result"]


def _run_stage(
    run: Path, stage: str, config: dict, state: dict, lease: int
) -> tuple[int, str | None]:
    process, owned, failure = None, {}, None
    limits = config["resources"]
    previous = state.get("resource_baseline")
    recent = state.get("resource_history", {}).get("last")
    if (
        previous is not None
        and recent is not None
        and recent["boot"] == previous["boot"]
        and recent["monotonic"] >= previous["monotonic"]
    ):
        previous = recent
    try:
        if "_resource_policy" in config:
            baseline = state.get("resource_baseline")
            if baseline is None or baseline.get("attempt") != state.get("execution_attempt"):
                return -1, "resource_wait:missing_attempt_baseline"
            try:
                fresh = _resource_sample(run, _process_table()[os.getpid()][1])
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                return -1, f"resource_wait:measurement_unavailable:{error}"
            if _calendar_expired(config, state):
                return -1, "wall_limit"
            if (
                fresh["monotonic"] - previous["monotonic"]
                > RESOURCE_POLICY["maximum_sample_age_seconds"]
            ):
                admission = _resource_admission(run, config, state)
                if not admission["safe"]:
                    return -1, "requested_stop" if admission[
                        "reason"
                    ] == "requested_stop" else "resource_wait:" + admission["reason"]
                fresh = admission["samples"][-1]
            else:
                condition = _resource_condition(fresh, previous, config, admission=True)
                if condition != "safe":
                    state.setdefault("resource_history", {})["last"] = fresh
                    return -1, "resource_wait:" + condition
            previous = fresh
        process = _launch(
            run,
            [
                sys.executable,
                "-m",
                MODULE,
                "stage",
                str(run.relative_to(ROOT)),
                stage,
                "--lease-fd",
                str(lease),
                *(
                    ["--pause-after-new-games", str(state["probe_new_games"])]
                    if state.get("probe_new_games") is not None
                    else []
                ),
                *(
                    ["--pause-after-updates", str(state["probe_updates"])]
                    if stage == "train" and state.get("probe_updates") is not None
                    else []
                ),
                *(
                    ["--pause-after-arena-games", str(state["probe_arena_games"])]
                    if stage == "arena" and state.get("probe_arena_games") is not None
                    else []
                ),
            ],
            f"{stage}-{state['retries'].get(stage, 0)}.log",
            lease,
        )
        state.update(stage_pid=process.pid, stage_identity=read_process_identity(process.pid))
        atomic(run / "state.json", encoded(state))
        signature, last_progress = None, time.monotonic()
        strikes = 0
        while True:
            rows = _stage_group_snapshot(process, state["stage_identity"])
            if rows[process.pid][2].startswith("Z"):
                break
            if (run / "STOP").exists():
                failure = "requested_stop"
                break
            current = _progress_signature(run, stage)
            if current != signature:
                signature, last_progress = current, time.monotonic()
            rss, owned = _sample_owned(process, owned)
            if "_resource_policy" in config:
                try:
                    sample = _resource_sample(run, rss + _process_table()[os.getpid()][1])
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    failure = f"resource_wait:measurement_unavailable:{error}"
                    break
                condition = _resource_condition(sample, previous, config)
                if (
                    "_dataset_admission" not in config
                    and "round" not in config
                    and condition == "safe"
                    and previous is not None
                    and sample["swap_bytes"] - state["resource_baseline"]["swap_bytes"]
                    > limits["maximum_swap_growth_gib"] * 1024**3
                    and (sample["swap_bytes"] - previous["swap_bytes"])
                    / (sample["monotonic"] - previous["monotonic"])
                    > RESOURCE_POLICY["quiet_bytes_per_second"]
                ):
                    condition = "sustained_attempt_growth"
                strikes = strikes + 1 if condition != "safe" else 0
                previous = sample
                history = state.setdefault("resource_history", {})
                for key in ("swap_bytes", "rss_bytes", "swapouts"):
                    history["peak_" + key] = max(history.get("peak_" + key, 0), sample[key])
                history["last"] = sample
                history["attempt_growth_bytes"] = (
                    sample["swap_bytes"] - state["resource_baseline"]["swap_bytes"]
                )
                if (
                    condition.startswith("critical:")
                    or strikes >= RESOURCE_POLICY["sustained_samples"]
                ):
                    failure = "resource_wait:" + condition
                free, swap = sample["free_bytes"], sample["swap_bytes"]
            else:
                sample = None
                free, swap = shutil.disk_usage(run).free, _swap_bytes()
            memory_free = (
                sample["memory_free_percent"]
                if sample
                else _memory_free_percent()
                if "minimum_memory_free_percent" in limits
                else None
            )
            metric = {
                "memory_free_percent": memory_free,
                "at": time.time(),
                "stage": stage,
                "stage_pid": process.pid,
                "rss_bytes": rss,
                "free_bytes": free,
                "swap_bytes": swap,
                "progress": signature,
                **(
                    {"resource_sample": sample, "condition": condition, "strikes": strikes}
                    if sample
                    else {}
                ),
            }
            with inside(run / "monitor.jsonl", exists=False).open("ab") as monitor:
                monitor.write(encoded(metric) + b"\n")
            state["owned_processes"] = {str(pid): identity for pid, identity in owned.items()}
            atomic(run / "state.json", encoded(state))
            if _calendar_expired(config, state):
                failure = "wall_limit"
            elif sample is not None:
                pass  # The reviewed multi-signal policy above owns resource decisions.
            elif memory_free is not None and memory_free < limits["minimum_memory_free_percent"]:
                failure = "memory_pressure_limit"
            elif rss > limits["maximum_process_rss_gib"] * 1024**3:
                failure = "memory_limit"
            elif free < limits["free_space_floor_gib"] * 1024**3:
                failure = "space_limit"
            elif swap - state["initial_swap_bytes"] > limits["maximum_swap_growth_gib"] * 1024**3:
                failure = "swap_limit"
            if (
                not failure
                and "_dataset_admission" not in config
                and time.monotonic() - last_progress > limits["stalled_seconds"]
            ):
                failure = "no_actual_progress"
            if failure:
                break
            time.sleep(
                RESOURCE_POLICY["interval_seconds"]
                if sample
                else limits["monitor_interval_seconds"]
            )
    except (OSError, ValueError, RuntimeError, sqlite3.Error, subprocess.SubprocessError) as error:
        _remember_error(state, "stage_spawn" if process is None else "supervision", error)
        if process is None:
            state["startup_failure"] = "before_spawn"
        failure = f"supervision_error:{type(error).__name__}:{error}"
    finally:
        if process is not None:
            if failure:
                with contextlib.suppress(OSError, ValueError):
                    stop(run)
            try:
                resource_stop = (failure or "").startswith("resource_wait:")
                grace = 20 if failure else 0
                if "_resource_policy" in config and (resource_stop or failure == "requested_stop"):
                    grace = (
                        0
                        if (failure or "").startswith("resource_wait:critical:")
                        else RESOURCE_POLICY["cooperative_stop_seconds"]
                    )
                cleanup = _cleanup(process, owned, grace_seconds=grace)
                state["cleanup"] = cleanup
                if cleanup["remaining_processes"]:
                    failure = "owned_process_cleanup_incomplete"
                else:
                    _mark_group_cleaned(run, process)
            except (
                OSError,
                ValueError,
                RuntimeError,
                sqlite3.Error,
                subprocess.SubprocessError,
            ) as error:
                _remember_error(state, "cleanup", error)
                failure = f"cleanup_error:{type(error).__name__}:{error}"
    if process is not None:
        supervised_interrupt = (
            "_resource_policy" in config
            and stage == "generate"
            and process.returncode in (-signal.SIGTERM, -signal.SIGKILL)
            and process.pid in state.get("cleanup", {}).get("signalled_pids", [])
            and not state.get("errors")
        )
        if supervised_interrupt:
            try:
                snapshot = _resume_snapshot(run, config)
                state["generation"] = {key: snapshot[key] for key in ("games", "rows", "tasks")}
            except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
                _remember_error(state, "interrupted_snapshot", error)
                failure = "interrupted_snapshot_failed"
                supervised_interrupt = False
        if (
            (failure == "requested_stop" or (failure or "").startswith("resource_wait:"))
            and process.returncode not in (0, PAUSED_EXIT)
            and not supervised_interrupt
        ):
            failure = f"stage_exit_{process.returncode}_during_stop"
        elif failure is None and process.returncode == PAUSED_EXIT and (run / "STOP").exists():
            failure = "requested_stop"
    return (
        process.returncode if process is not None and process.returncode is not None else -1,
        failure,
    )


def _allocation_task(run: Path, stage: str) -> str:
    resume = _json(run / "fit/resume.json") if (run / "fit/resume.json").exists() else {"step": 0}
    return f"{stage}:{resume['step']}"


def _wait_allocation(run: Path, config: dict, state: dict, stage: str) -> bool:
    """Wait without touching a sampler; retry budgets survive attempts and explicit resumes."""
    path = run / "allocation-retries.json"
    counters = _json(path) if path.exists() else {}
    failures = counters.get(_allocation_task(run, stage), 0)
    if not failures:
        return True
    event = _json(run / "allocation-failure.json")
    state.update(
        status="running",
        reason="resource_wait",
        resource_wait={
            **event,
            "failures": failures,
            "automatic_retries_remaining": max(0, 4 - failures),
        },
    )
    atomic(run / "state.json", encoded(state))
    while not (run / "STOP").exists():
        time.sleep(15)
        try:
            sample = _resource_sample(run, _process_table()[os.getpid()][1])
        except (OSError, ValueError, subprocess.SubprocessError):
            continue
        # Confirm recovery before consuming another finite task attempt. Host pressure
        # can delay a retry after actual OOM, but never kills a running computation.
        baseline = event.get("resource_sample") or {}
        recovered = sample["memory_free_percent"] >= baseline.get(
            "memory_free_percent", 100
        ) + 5 or (baseline.get("pressure") in (2, 4) and sample["pressure"] == 1)
        if (
            failures <= 3
            and time.time() - event["at"] >= 60
            and recovered
            and _resource_condition(sample, None, config, admission=True) == "safe"
        ):
            return True
    return False


def work(
    run: Path,
    inherited_fd: int | None = None,
    *,
    pause_after_new_games: int | None = None,
    pause_after_arena_games: int | None = None,
    pause_after_updates: int | None = None,
) -> dict:
    with _lease(run, inherited_fd) as lease:
        config = verify(run)
        _require_migration(run, config)
        residual = _residual_stage_group(run)
        if residual is not None:
            state = _json(run / "state.json")
            state.update(
                status="needs_astra",
                reason="residual_stage_group_unproven",
                residual_group=residual,
            )
            atomic(run / "state.json", encoded(state))
            return state
        prior = _state(run)
        if prior["status"] in {"needs_astra", "awaiting_astra_review", "awaiting_astra_browser"}:
            return prior
        began = prior.get("began_at", time.time())
        _number(
            began,
            0,
            math.inf if "_calendar_policy" in config else time.time(),
            "began_at",
            integer=False,
        )
        initial_swap = prior.get("initial_swap_bytes")
        initial_swap = _swap_bytes() if initial_swap is None else initial_swap
        _number(initial_swap, 0, 1024**5, "initial_swap_bytes")
        retries = prior.get("retries", {})
        if not isinstance(retries, dict) or any(stage not in _stages(config) for stage in retries):
            raise ValueError("invalid persisted retries")
        for count in retries.values():
            _number(count, 0, config["resources"]["maximum_retries"], "retry count")
        state = {
            "schema": "open_shogiai_evaluator_state/v2"
            if config["generation"].get("defense_campaign")
            else "open_shogiai_evaluator_state/v1",
            "run_id": config["run_id"],
            "run_sha256": digest(run / "run.json"),
            "pid": os.getpid(),
            "process_identity": read_process_identity(os.getpid()),
            "began_at": began,
            "initial_swap_bytes": initial_swap,
            "retries": retries,
            "status": "running",
        }
        for key in (
            "operation_revision",
            "execution_attempt",
            "resource_baseline",
            "resource_history",
        ):
            if key in prior:
                state[key] = prior[key]
        if pause_after_updates is not None:
            _number(pause_after_updates, 1, 32, "initial training update prefix")
            state["probe_updates"] = pause_after_updates
        if pause_after_new_games is not None:
            _number(pause_after_new_games, 1, 12, "probe trajectory bound")
            state["probe_new_games"] = pause_after_new_games
        if pause_after_arena_games is not None:
            _number(pause_after_arena_games, 2, 36, "arena completed-task pause bound")
            state["probe_arena_games"] = pause_after_arena_games
        if state["process_identity"] is None:
            raise RuntimeError("supervisor process identity unavailable")
        previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        for sig in previous_handlers:
            signal.signal(sig, lambda _signum, _frame: stop(run))
        try:
            if "_resource_policy" in config:
                attempt = _json(run / "attempts" / f"{state['execution_attempt']:06d}.json")
                if state.get("resource_baseline") != attempt.get("resource_baseline"):
                    raise ValueError("attempt resource baseline changed")
            atomic(run / "state.json", encoded(state))
            for stage in _stages(config):
                if pause_after_new_games is not None and stage != "generate":
                    state.update(status="ready_for_luna", reason="generation_probe_complete")
                    break
                verify(run)
                if (run / "STOP").exists():
                    state.update(status="stopped", reason="requested_stop")
                    break
                if (run / f"{stage}-complete.json").exists():
                    _verify_completion(run, stage, config)
                    continue
                while True:
                    state.update(stage=stage, status="running", stage_started_at=time.time())
                    state.pop("reason", None)
                    state.pop("startup_failure", None)
                    atomic(run / "state.json", encoded(state))
                    if (
                        "_dataset_admission" in config or "round" in config
                    ) and not _wait_allocation(run, config, state, stage):
                        state.update(status="stopped", reason="requested_stop")
                        break
                    returncode, failure = _run_stage(run, stage, config, state, lease)
                    if (
                        returncode == RESOURCE_EXIT
                        and failure is None
                        and ("_dataset_admission" in config or "round" in config)
                    ):
                        counters_path = run / "allocation-retries.json"
                        counters = _json(counters_path) if counters_path.exists() else {}
                        task = _allocation_task(run, stage)
                        counters[task] = counters.get(task, 0) + 1
                        atomic(counters_path, encoded(counters))
                        continue
                    if failure or returncode:
                        if (
                            failure is None
                            and returncode in (-signal.SIGTERM, -signal.SIGINT)
                            and retries.get(stage, 0) < config["resources"]["maximum_retries"]
                        ):
                            retries[stage] = retries.get(stage, 0) + 1
                            atomic(run / "state.json", encoded(state))
                            continue
                        state.update(
                            status="stopped"
                            if failure == "requested_stop"
                            or (failure or "").startswith("resource_wait:")
                            else "needs_astra",
                            reason=failure or f"stage_exit_{returncode}",
                        )
                        break
                    if pause_after_new_games is not None and stage == "generate":
                        pause = _json(run / "generation-probe.json")
                        if pause["run_sha256"] != digest(run / "run.json"):
                            raise ValueError("probe receipt belongs to another run")
                        state.update(
                            status="ready_for_luna",
                            reason="generation_probe_complete",
                            generation=pause["result"],
                        )
                        break
                    if (
                        pause_after_updates is not None
                        and stage == "train"
                        and not (run / "train-complete.json").exists()
                    ):
                        state.update(status="ready_for_luna", reason="training_prefix_complete")
                        break
                    if pause_after_arena_games is not None and stage == "arena":
                        arena = _json(run / "arena/arena.json")
                        if arena["status"] == "paused":
                            _candidate_review(run)
                            state.update(status="ready_for_luna", reason="arena_probe_complete")
                            break
                    _verify_completion(run, stage, config)
                    break
                if state["status"] != "running":
                    break
            else:
                if config["generation"].get("defense_campaign"):
                    review = _candidate_review(run)
                    state.update(
                        review_sha256=digest(run / "candidate-review.json"),
                        meets_frozen_criteria=review["meets_frozen_criteria"],
                    )
                state.update(
                    status="awaiting_astra_review"
                    if config["generation"].get("defense_campaign")
                    else "awaiting_astra_browser",
                    stage="complete",
                    completed_at=time.time(),
                )
        except (
            OSError,
            ValueError,
            RuntimeError,
            KeyError,
            sqlite3.Error,
            subprocess.SubprocessError,
        ) as error:
            _remember_error(state, "work", error)
            state.update(status="needs_astra", reason=f"{type(error).__name__}:{error}")
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
            if state.get("execution_attempt"):
                try:
                    atomic(
                        run / "attempts" / f"{state['execution_attempt']:06d}-result.json",
                        encoded(state),
                    )
                except OSError as error:
                    _remember_error(state, "attempt_result", error)
                    state.update(status="needs_astra", reason="attempt_result_write_failed")
            _save_state(run, state)
        return state


def _arena_config(run: Path, config: dict) -> dict:
    generation = config["generation"]
    model = selected_model(run)
    return {
        **config["evaluation"],
        "seed": config["seed"],
        "probe_path": config["runtime"]["probe"]["path"],
        "probe_sha256": config["runtime"]["probe"]["sha256"],
        "baseline_path": generation["leaf_path"],
        "baseline_sha256": generation["leaf_sha256"],
        "candidate_path": str(model.relative_to(ROOT)),
        "candidate_sha256": digest(model),
        "dataset_path": str((run / "data" / "dataset").relative_to(ROOT)),
    }


def _candidate_review(run: Path) -> dict:
    """Combine predeclared screens without mistaking an arena win for overall adoption."""
    offline = _json(run / "development-test.json")
    arena = _json(run / "arena/arena.json")
    groups = offline["groups"]
    is_round = "round" in _json(run / "run.json")
    baseline = "baseline" if is_round else "r3"
    checked_groups = (
        ("general", "opening", "defense", "attack_end") if is_round else ("general", "attack_end")
    )
    scalar_preserved = all(
        groups["candidate"][g]["loss"] <= groups[baseline][g]["loss"] * 1.03 for g in checked_groups
    )
    screen = offline["move_quality_screen"]
    screen_pass = screen["screen_pass"]
    arena_pass = arena["adoption_criteria_met"] is True
    result = {
        "schema": "open_shogiai_candidate_review/v1",
        "run_sha256": digest(run / "run.json"),
        "candidate_sha256": digest(selected_model(run)),
        "offline_sha256": digest(run / "development-test.json"),
        "arena_sha256": digest(run / "arena/arena.json"),
        "scalar_all_groups_preserved"
        if is_round
        else "scalar_general_attack_preserved": scalar_preserved,
        "move_quality_screen_pass": screen_pass,
        "move_quality_screen_status": screen.get("status", "complete"),
        "arena_complete": arena.get("summary", {}).get("all_planned_complete", False),
        "runtime_audit_pass": _json(run / "model-audit.json").get("status") == "PASS"
        if (run / "model-audit.json").exists()
        else False,
        "development_candidate_available": (run / "audit-complete.json").exists(),
        "baseline_arena_pass" if is_round else "r3_arena_pass": arena_pass,
        "scalar_checked_groups": list(checked_groups),
        "optional_screen_required": not is_round,
        "meets_frozen_criteria": bool(
            scalar_preserved and (is_round or screen_pass) and arena_pass
        ),
        "promotion_performed": False,
        "decision_owner": "user after fixed comparison and playable development registration"
        if is_round
        else "Astra",
        "human_shodan_validated": False,
    }
    atomic(run / "candidate-review.json", encoded(result))
    return result


def stage_run(
    run: Path,
    stage: str,
    inherited_fd: int | None = None,
    *,
    pause_after_new_games: int | None = None,
    pause_after_arena_games: int | None = None,
    pause_after_updates: int | None = None,
) -> dict:
    if stage not in ALL_STAGES:
        raise ValueError("unknown stage")
    with _lease(run, inherited_fd):
        config = verify(run)
        _require_migration(run, config)
        state = _state(run)
        if state["status"] in {"needs_astra", "awaiting_astra_review", "awaiting_astra_browser"}:
            raise ValueError(
                "stage execution requires Astra review; terminal run is not restartable"
            )
        if (run / "STOP").exists():
            raise InterruptedError("stop requested before stage launch")
        if (run / f"{stage}-complete.json").exists():
            return _verify_completion(run, stage, config)
        if config["generation"].get("defense_campaign"):
            for previous in _stages(config)[: _stages(config).index(stage)]:
                _verify_completion(run, previous, config)
        _register_stage_group(run, stage)
        generation = config["generation"]
        if stage == "generate" and generation.get("prepared_dataset"):
            from .r4_data import verify_dataset

            ref = generation["prepared_dataset"]
            dataset = verify_dataset(ROOT, inside(ref["path"]), ref["manifest_sha256"])
            atomic(run / "data/generation.json", encoded(generation))
            result = {
                "status": "complete",
                "mode": "verified_existing_inputs",
                "dataset_sha256": ref["manifest_sha256"],
                "source_rows": dataset["source_rows"],
            }
            atomic(run / "data/generation-complete.json", encoded(result))
        elif stage == "generate":
            if pause_after_new_games is not None:
                _number(pause_after_new_games, 1, 12, "probe trajectory bound")
                result = generate(
                    ROOT, run / "data", generation, pause_after_new_games=pause_after_new_games
                )
                verify(run)
                atomic(
                    run / "generation-probe.json",
                    encoded({"run_sha256": digest(run / "run.json"), "result": result}),
                )
                return result
            result = generate(ROOT, run / "data", generation)
            if generation.get("recovery_policy"):
                if result.get("status") != "complete":
                    raise ValueError(
                        "finite generation coverage exhausted: " + str(result.get("status"))
                    )
                _prepare_recovery(run, config)
                result = _json(run / "data/generation-complete.json")
            if not generation.get("recovery_policy") and result.get("games") != generation["games"]:
                raise ValueError("generation stopped before the planned trajectory count")
        elif stage == "prepare":
            if generation.get("prepared_dataset"):
                from .r4_data import import_dataset

                result = import_dataset(ROOT, run / "data", generation)
            else:
                result = prepare(
                    ROOT, run / "data", generation, config["excluded_development_sfens"]
                )
            if generation.get("recovery_policy"):
                result = _dataset(run, config)
        else:
            _dataset(run, config)
            if stage == "train":
                from .evaluator_training import train

                result = train(
                    run / "data" / "dataset",
                    run / "fit",
                    config["training"],
                    _training_identity(run, config),
                    stop_after=pause_after_updates,
                )
                if result["status"] == "paused_for_resume_check":
                    verify(run)
                    return result
                if result["status"] != "complete":
                    raise InterruptedError("training stopped with coherent resume checkpoint")
            elif stage == "audit":
                import torch

                from .evaluator_training import arrays, evaluate, evaluate_groups, grouped_arrays
                from .phase10v_model import Phase10VModel, torch_parameters

                _training_summary(run, config)
                torch.set_num_threads(config["training"]["threads"])
                model, report = selected_model(run), run / "model-audit.json"
                if not report.exists():
                    subprocess.run(
                        [
                            "node",
                            str(ROOT / "scripts/check_evaluator_model.mjs"),
                            str(inside(config["runtime"]["module"]["path"])),
                            str(model),
                            digest(model),
                            str(inside(config["runtime"]["replay"]["path"])),
                            str(report),
                        ],
                        cwd=ROOT,
                        check=True,
                        timeout=180,
                    )
                _audit_report(run, config)
                test = arrays(run / "data" / "dataset", "development_test")
                metrics = {
                    name: evaluate(
                        torch_parameters(Phase10VModel.read(path)),
                        test,
                        config["training"]["batch_size"],
                    )
                    for name, path in (
                        ("baseline", inside(generation["leaf_path"])),
                        ("candidate", model),
                    )
                }
                if generation.get("defense_campaign"):
                    from .defense_evaluation import screen

                    grouped = grouped_arrays(run / "data" / "dataset", "development_test")
                    metrics["groups"] = {
                        name: evaluate_groups(
                            torch_parameters(Phase10VModel.read(path)),
                            grouped,
                            config["training"]["batch_size"],
                        )
                        for name, path in (
                            (
                                "baseline" if "round" in config else "r3",
                                inside(generation["leaf_path"]),
                            ),
                            ("candidate", model),
                        )
                    }
                    metrics["move_quality_screen"] = screen(ROOT, run, config, model=model)
                atomic(run / "development-test.json", encoded(metrics))
                result = {
                    "model_sha256": digest(model),
                    "audit_sha256": digest(report),
                    "unknown_development_metrics": metrics,
                }
            elif stage == "integrate":
                from .evaluator_development import register

                _training_summary(run, config)
                _audit_report(run, config)
                result = register(ROOT, run, config, selected_model(run), run / "development")
            else:
                from .evaluator_arena import run_arena

                _training_summary(run, config)
                _audit_report(run, config)
                result = run_arena(
                    ROOT,
                    run / "arena",
                    _arena_config(run, config),
                    pause_after_games=pause_after_arena_games,
                )
                if result["status"] == "paused":
                    verify(run)
                    return result  # No arena completion receipt for a partial schedule.
                _arena_complete(result, 32 + config["evaluation"]["startpos_demonstration_games"])
        verify(run)
        if (run / "STOP").exists():
            raise InterruptedError("stop requested before completion publication")
        receipt = {
            "schema": "open_shogiai_evaluator_stage/v1",
            "stage": stage,
            "run_sha256": digest(run / "run.json"),
            "result": result,
            "artifacts": _completion_artifacts(run, stage),
        }
        atomic(run / f"{stage}-complete.json", encoded(receipt))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "seal",
            "approve-operations",
            "migrate",
            "start",
            "resume",
            "pause",
            "probe",
            "stop",
            "status",
            "diagnose",
            "rehearsal",
            "work",
            "stage",
        ),
    )
    parser.add_argument("path")
    parser.add_argument("stage", nargs="?", choices=ALL_STAGES)
    parser.add_argument("--lease-fd", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--pause-after-new-games", type=int, default=None)
    parser.add_argument("--pause-after-arena-games", type=int, default=None)
    parser.add_argument("--pause-after-updates", type=int, default=None)
    parser.add_argument("--abolish-calendar-limit", action="store_true")
    parser.add_argument("--admit-existing-data", action="store_true")
    parser.add_argument("--post-training-evaluation", action="store_true")
    args = parser.parse_args()
    path = inside(args.path)
    if args.abolish_calendar_limit and args.command != "approve-operations":
        parser.error("--abolish-calendar-limit is only valid with approve-operations")
    if args.admit_existing_data and args.command != "approve-operations":
        parser.error("--admit-existing-data is only valid with approve-operations")
    if args.post_training_evaluation and args.command != "approve-operations":
        parser.error("--post-training-evaluation is only valid with approve-operations")
    if args.command == "seal":
        result = seal(path)
    elif args.command == "approve-operations":
        result = _approve_operations(
            path,
            abolish_calendar_limit=args.abolish_calendar_limit,
            admit_existing_data=args.admit_existing_data,
            post_training_evaluation=args.post_training_evaluation,
        )
    elif args.command == "rehearsal":
        from .evaluator_development import rehearsal

        with _lease(path):
            revision = None
            if (path / "approved-operation.json").exists():
                ref = _json(path / "approved-operation.json")
                revision_path = inside(ref["path"])
                if revision_path.parent != path / "operations":
                    raise ValueError("operation revision outside run")
                _reference(revision_path, ref["sha256"])
                revision = _json(revision_path)
            config = verify(path, operation_revision=revision)
            if (
                _state(path)["status"] != "ready_for_luna"
                or _residual_stage_group(path) is not None
            ):
                raise ValueError("rehearsal requires a normally paused initial prefix")
            result = rehearsal(ROOT, path, config)
    elif args.command == "diagnose":
        result = diagnose(path)
    elif args.command == "migrate":
        result = migrate(path)
    elif args.command == "stage":
        try:
            result = stage_run(
                path,
                args.stage,
                args.lease_fd,
                pause_after_new_games=args.pause_after_new_games,
                pause_after_arena_games=args.pause_after_arena_games,
                pause_after_updates=args.pause_after_updates,
            )
        except (MemoryError, RuntimeError) as error:
            from .evaluator_training import TrainingResourceWaitError, allocation_failure

            if not isinstance(error, TrainingResourceWaitError) and not allocation_failure(error):
                raise
            try:
                sample = _resource_sample(path, _process_table()[os.getpid()][1])
            except (OSError, ValueError, subprocess.SubprocessError):
                sample = None
            atomic(
                path / "allocation-failure.json",
                encoded(
                    {
                        "at": time.time(),
                        "stage": args.stage,
                        "reason": "allocation_failure",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "resource_sample": sample,
                        "checkpoint": _json(path / "fit/resume.json")
                        if (path / "fit/resume.json").exists()
                        else None,
                    }
                ),
            )
            raise SystemExit(RESOURCE_EXIT) from None
        except InterruptedError as error:
            if (
                error.errno is not None
                or str(error)
                not in {
                    "requested stop; completed trajectories retained",
                    "stop requested before stage launch",
                    "training stopped with coherent resume checkpoint",
                    "arena stopped with retained game receipts",
                    "move screen stopped",
                    "stop requested before completion publication",
                }
                or not (path / "STOP").exists()
            ):
                raise
            print(json.dumps({"status": "paused", "reason": str(error)}), flush=True)
            raise SystemExit(PAUSED_EXIT) from None
    elif args.command == "work":
        result = work(
            path,
            args.lease_fd,
            pause_after_new_games=args.pause_after_new_games,
            pause_after_arena_games=args.pause_after_arena_games,
            pause_after_updates=args.pause_after_updates,
        )
    elif args.command == "probe":
        result = start(path, pause_after_new_games=args.pause_after_new_games or 1)
    elif args.command == "resume":
        result = start(
            path,
            pause_after_new_games=args.pause_after_new_games,
            recover_startup=True,
            pause_after_arena_games=args.pause_after_arena_games,
            pause_after_updates=args.pause_after_updates,
        )
    elif args.command == "pause":
        result = pause(path)
    else:
        result = {"start": start, "stop": stop, "status": status}[args.command](path)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
