"""Durable, fail-closed execution boundary for the frozen Phase 10R campaign.

The Phase 10R documents freeze a new OSAVAL02 evaluator and its cross-runtime
contract.  This module owns campaign receipts and refuses to substitute the
older OSAVAL01 model when the required backend is absent.  It is intentionally
small: execution infrastructure may not redefine the frozen model or gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from open_shogi_training.data.phase10r_registry import load_phase10r_registry
from open_shogi_training.phase10r import (
    EXPECTED_ARENA_GAMES,
    Phase10RValidationError,
    memory_estimates,
    run_micro_overfit,
    run_pipeline_sanity,
    validate_phase10r,
)

RUNS_DIRECTORY: Final = Path("local/phase10r-runs")
FROZEN_BRANCH: Final = "codex/phase10r-curriculum"
SCALE_VALUES: Final = ("1m", "10m", "50m", "100m", "500m", "1b")
TRAIN_VARIANTS: Final = (
    "sparse-pair-policy-wdl",
    "factorized-pair-triple-policy-score",
)
BACKEND_GAP: Final = (
    "OSAVAL02 sparse pair/triple training and Rust/Wasm incremental runtime are not "
    "implemented; the available backend is OSAVAL01 only."
)
LEAKAGE_GAP: Final = (
    "full approved-source canonical/history split-leakage scan has not been run; the "
    "data-foundation report records overlap as unavailable until canonical join."
)


class Phase10RRunError(RuntimeError):
    """Raised when the frozen execution boundary cannot be satisfied."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _hash_manifest(root: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    path = root / "configs/phase10r/frozen-controls.sha256"
    for line in path.read_text(encoding="ascii").splitlines():
        if not line:
            continue
        digest, separator, name = line.partition("  ")
        if separator != "  " or len(digest) != 64 or name in entries:
            raise Phase10RRunError("frozen hash manifest is malformed")
        entries[name] = digest
    return entries


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    try:
        with path.open("xb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
    except FileExistsError as error:
        raise Phase10RRunError(f"refusing to overwrite immutable receipt: {path}") from error


def _append_event(root: Path, event: dict[str, Any]) -> None:
    path = root / RUNS_DIRECTORY / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def _receipt_path(root: Path, command: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return root / RUNS_DIRECTORY / f"{stamp}-{command}.json"


def _base_receipt(root: Path, command: str, argv: Sequence[str]) -> dict[str, Any]:
    try:
        commit = _git(root, "rev-parse", "HEAD")
        dirty = _git(root, "status", "--porcelain")
        branch = _git(root, "branch", "--show-current")
    except (OSError, subprocess.CalledProcessError) as error:
        commit = None
        dirty = None
        branch = None
        git_error = str(error)
    else:
        git_error = None
    return {
        "schema": "open_shogi_ai_phase10r_run_receipt/v1",
        "command": command,
        "argv": list(argv),
        "started_at_utc": _utc_now(),
        "repository": root.name,
        "git": {"branch": branch, "commit": commit, "dirty": dirty, "error": git_error},
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "device": platform.machine(),
            "pid": os.getpid(),
        },
    }


def _finish_receipt(
    root: Path,
    command: str,
    argv: Sequence[str],
    receipt: dict[str, Any],
) -> Path:
    receipt["ended_at_utc"] = _utc_now()
    rss_unit = 1 if sys.platform == "darwin" else 1024
    receipt["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_unit
    path = _receipt_path(root, command)
    _write_new_json(path, receipt)
    _append_event(
        root,
        {
            "schema": "open_shogi_ai_phase10r_event/v1",
            "event": "receipt_written",
            "timestamp_utc": receipt["ended_at_utc"],
            "command": command,
            "receipt": str(path.relative_to(root)),
            "receipt_sha256": _sha256(path),
            "argv": list(argv),
        },
    )
    return path


def _disk_receipt(root: Path) -> dict[str, Any]:
    data_root = Path(os.environ.get("OPENSHOGI_DATA_ROOT", root / "local/phase10r-data"))
    if not data_root.is_absolute():
        data_root = root / data_root
    usage = shutil.disk_usage(data_root if data_root.exists() else root)
    minimum = 150 * 1024**3
    try:
        display_path = str(data_root.relative_to(root))
    except ValueError:
        display_path = "<external-data-root>"
    return {
        "path": display_path,
        "free_bytes": usage.free,
        "minimum_free_bytes": minimum,
        "passed": usage.free >= minimum,
    }


def _thermal_receipt() -> dict[str, Any]:
    if sys.platform != "darwin":
        return {"status": "not_applicable", "passed": True}
    try:
        completed = subprocess.run(
            ["pmset", "-g", "therm"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "unavailable", "passed": False, "error": str(error)}
    output = completed.stdout.strip()
    return {
        "status": "observed" if completed.returncode == 0 else "error",
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "summary_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
    }


def _backend_receipt(root: Path) -> dict[str, Any]:
    neural = root / "engine/core/src/neural.rs"
    model_module = root / "training/open_shogi_training/phase10r_model.py"
    osaval02 = neural.is_file() and b"OSAVAL02" in neural.read_bytes()
    return {
        "required_format": "OSAVAL02",
        "osaval02_parser_present": osaval02,
        "phase10r_model_module_present": model_module.is_file(),
        "available_legacy_format": "OSAVAL01",
        "pure_runtime_configured": True,
        "passed": osaval02 and model_module.is_file(),
        "stop_reason": None if osaval02 and model_module.is_file() else BACKEND_GAP,
    }


def _run_preflight(root: Path, argv: Sequence[str]) -> tuple[dict[str, Any], bool]:
    receipt = _base_receipt(root, "preflight", argv)
    failures: list[str] = []
    try:
        validate_phase10r(root)
        receipt["frozen_validation"] = "passed"
        receipt["frozen_hashes"] = _hash_manifest(root)
    except (Phase10RValidationError, OSError, KeyError) as error:
        receipt["frozen_validation"] = "failed"
        failures.append(f"frozen validation: {error}")
    for name, function in (
        ("pipeline_sanity", lambda: run_pipeline_sanity(root)),
        ("micro_overfit", run_micro_overfit),
        ("memory", lambda: memory_estimates(root)),
    ):
        try:
            receipt[name] = function()
        except (Phase10RValidationError, OSError, RuntimeError, ValueError) as error:
            receipt[name] = {"status": "failed", "error": str(error)}
            failures.append(f"{name}: {error}")
    try:
        registry = load_phase10r_registry(root / "configs/phase10r/source-registry.yaml")
        receipt["source_registry"] = {
            "artifact_count": len(registry.artifacts),
            "approved_training_artifacts": len(registry.approved_artifacts()),
            "pending_or_denied_admitted": False,
        }
    except (OSError, ValueError, KeyError) as error:
        receipt["source_registry"] = {"status": "failed", "error": str(error)}
        failures.append(f"source registry: {error}")
    receipt["disk"] = _disk_receipt(root)
    if not receipt["disk"]["passed"]:
        failures.append("disk free space is below the frozen 150 GiB floor")
    receipt["thermal"] = _thermal_receipt()
    if not receipt["thermal"]["passed"]:
        failures.append("thermal status could not be observed")
    receipt["backend"] = _backend_receipt(root)
    if not receipt["backend"]["passed"]:
        failures.append(BACKEND_GAP)
    receipt["split_leakage"] = {
        "status": "not_run",
        "passed": False,
        "reason": LEAKAGE_GAP,
    }
    failures.append(LEAKAGE_GAP)
    receipt["frozen_runtime"] = {
        "mode": "PureValue",
        "handcrafted_leaf_contribution": False,
        "third_party_weights": False,
        "passed": True,
    }
    receipt["status"] = "passed" if not failures else "blocked"
    receipt["exit_status"] = 0 if not failures else 2
    receipt["failures"] = failures
    return receipt, not failures


def _write_receipt_and_return(
    root: Path, command: str, argv: Sequence[str], receipt: dict[str, Any]
) -> int:
    path = _finish_receipt(root, command, argv, receipt)
    print(json.dumps({"receipt": str(path), "status": receipt["status"]}, sort_keys=True))
    return int(receipt.get("exit_status", 0))


def _blocked_operation(root: Path, command: str, argv: Sequence[str], reason: str) -> int:
    receipt = _base_receipt(root, command, argv)
    receipt.update({"status": "blocked", "exit_status": 2, "stop_reason": reason})
    return _write_receipt_and_return(root, command, argv, receipt)


def _write_text_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
    except FileExistsError as error:
        raise Phase10RRunError(f"refusing to overwrite immutable report: {path}") from error


def _report(root: Path, argv: Sequence[str], scale: str) -> int:
    receipt = _base_receipt(root, "report", argv)
    runs = root / RUNS_DIRECTORY
    prior_receipts: list[dict[str, Any]] = []
    if runs.exists():
        for path in sorted(runs.glob("*.json")):
            try:
                prior_receipts.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
    stop_reasons = sorted(
        {
            reason
            for item in prior_receipts
            for reason in item.get("failures", [])
            if isinstance(reason, str)
        }
    )
    stop_reasons.extend(
        item["stop_reason"]
        for item in prior_receipts
        if isinstance(item.get("stop_reason"), str) and item["stop_reason"] not in stop_reasons
    )
    status = "blocked" if stop_reasons else "no_execution_receipts"
    next_command = (
        "Implement and hash-bind OSAVAL02 plus the approved-source canonical/history leakage "
        "scan, then rerun preflight before prepare --scale 1m."
    )
    report_data = {
        "schema": "open_shogi_ai_phase10r_execution_report/v1",
        "status": status,
        "scale_requested": scale,
        "branch": receipt["git"]["branch"],
        "commit": receipt["git"]["commit"],
        "dirty": receipt["git"]["dirty"],
        "elapsed": "preflight only; no training or game process launched",
        "acquired_this_campaign": {},
        "consumed_examples": {},
        "deduplicated_examples": {},
        "teacher_labels_this_campaign": 0,
        "trained_variants": [],
        "arena_results": [],
        "bootstrapped_generations": [],
        "pure_selfplay_generations": [],
        "current_best_pure_candidate": None,
        "sol_review_gate_reached": False,
        "final_objective_passed": False,
        "promotion": {"overall_champion_mutated": False, "promotion_performed": False},
        "stop_reasons": stop_reasons,
        "receipts": [
            str(path.relative_to(root))
            for path in sorted(runs.glob("*.json"))
            if path.name != "PHASE10R_EXECUTION_REPORT.json"
        ],
        "next_exact_command": next_command,
    }
    json_report = runs / "PHASE10R_EXECUTION_REPORT.json"
    markdown_report = runs / "PHASE10R_EXECUTION_REPORT.md"
    _write_new_json(json_report, report_data)
    _write_text_new(
        markdown_report,
        "# Phase 10R-C execution report\n\n"
        f"Status: **{status}**\n\n"
        "No training, teacher labeling, Arena, cross-play, self-play, promotion, or holdout "
        "inspection was launched.\n\n"
        "## Stop reasons\n\n"
        + "\n".join(f"- {reason}" for reason in stop_reasons)
        + "\n\n## Next exact command\n\n"
        f"`{next_command}`\n",
    )
    receipt.update(
        {
            "status": status,
            "exit_status": 2 if status == "blocked" else 0,
            "scale": scale,
            "report_json": str(json_report.relative_to(root)),
            "report_markdown": str(markdown_report.relative_to(root)),
            "stop_reasons": stop_reasons,
        }
    )
    return _write_receipt_and_return(root, "report", argv, receipt)


def _parse_scale(value: str) -> str:
    if value not in SCALE_VALUES:
        raise argparse.ArgumentTypeError(f"scale must be one of {', '.join(SCALE_VALUES)}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--root", type=Path, default=Path("."))
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--root", type=Path, default=Path("."))
    prepare.add_argument("--scale", type=_parse_scale, required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--root", type=Path, default=Path("."))
    train.add_argument("--scale", type=_parse_scale, required=True)
    train.add_argument("--variant", choices=TRAIN_VARIANTS, required=True)
    train.add_argument("--resume", action="store_true")
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--root", type=Path, default=Path("."))
    evaluate.add_argument("--scale", type=_parse_scale, required=True)
    evaluate.add_argument("--all-source-held-out", action="store_true", required=True)
    evaluate.add_argument("--cross-runtime", action="store_true", required=True)
    evaluate.add_argument("--incremental-parity", action="store_true", required=True)
    select_hard = subparsers.add_parser("select-hard")
    select_hard.add_argument("--root", type=Path, default=Path("."))
    select_hard.add_argument("--scale", type=_parse_scale, required=True)
    label_hard = subparsers.add_parser("label-hard")
    label_hard.add_argument("--root", type=Path, default=Path("."))
    label_hard.add_argument("--scale", type=_parse_scale, required=True)
    label_hard.add_argument("--resume", action="store_true")
    report = subparsers.add_parser("report")
    report.add_argument("--root", type=Path, default=Path("."))
    report.add_argument("--scale", type=_parse_scale, required=True)
    arena = subparsers.add_parser("arena")
    arena.add_argument("--root", type=Path, default=Path("."))
    arena.add_argument("--gate", choices=tuple(EXPECTED_ARENA_GAMES), required=True)
    arena.add_argument("--resume", action="store_true")
    crossplay = subparsers.add_parser("crossplay")
    crossplay.add_argument("--root", type=Path, default=Path("."))
    crossplay.add_argument("--opponents", required=True)
    crossplay.add_argument("--resume", action="store_true")
    decisive = subparsers.add_parser("select-decisive-errors")
    decisive.add_argument("--root", type=Path, default=Path("."))
    label_decisive = subparsers.add_parser("label-decisive-errors")
    label_decisive.add_argument("--root", type=Path, default=Path("."))
    label_decisive.add_argument("--resume", action="store_true")
    selfplay = subparsers.add_parser("selfplay")
    selfplay.add_argument("--root", type=Path, default=Path("."))
    selfplay.add_argument("--games", type=int, required=True)
    selfplay.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    command = args.command
    command_argv = tuple(argv if argv is not None else sys.argv[1:])
    if command == "preflight":
        receipt, _ = _run_preflight(root, command_argv)
        return _write_receipt_and_return(root, command, command_argv, receipt)
    if command == "report":
        return _report(root, command_argv, args.scale)
    return _blocked_operation(root, command, command_argv, BACKEND_GAP)


if __name__ == "__main__":
    raise SystemExit(main())
