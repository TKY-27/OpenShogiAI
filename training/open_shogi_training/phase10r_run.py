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
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from open_shogi_training.data.phase10r_registry import load_phase10r_registry
from open_shogi_training.data.phase10r_scan import (
    Phase10RScanError,
    scan_phase10r_population,
)
from open_shogi_training.phase10r import (
    EXPECTED_ARENA_GAMES,
    Phase10RValidationError,
    memory_estimates,
    run_micro_overfit,
    run_pipeline_sanity,
    validate_phase10r,
)
from open_shogi_training.phase10r_campaign import (
    Phase10RCampaignError,
    evaluate_scale,
    label_hard,
    select_hard,
    train_variant,
)
from open_shogi_training.phase10r_execution import (
    Phase10RExecutionError,
    preparation_manifest_path,
    prepare_scale,
    validate_preparation,
)
from open_shogi_training.phase10r_lineage import (
    completed_teacher_bound_candidates,
    load_teacher_binding_control,
)
from open_shogi_training.phase10r_model import (
    VARIANT_PAIR,
    VARIANT_PRIMARY,
    expected_parameter_count,
    parse_osaval02,
)
from open_shogi_training.phase10r_training import (
    Phase10RExample,
    Phase10RModel,
    Phase10RTrainingError,
    TrainingConfig,
    export_osaval02_artifact,
    run_bounded_training,
)

RUNS_DIRECTORY: Final = Path("local/phase10r-runs")
FROZEN_BRANCH: Final = "codex/phase10r-curriculum"
SCALE_VALUES: Final = ("1m", "10m", "50m", "100m", "500m", "1b")
TRAIN_VARIANTS: Final = (
    "sparse-pair-policy-wdl",
    "factorized-pair-triple-policy-score",
)
BACKEND_GAP: Final = (
    "bounded Phase 10R training backend validation did not pass; no 1M or larger rung may start."
)
LEAKAGE_GAP: Final = (
    "approved-source canonical/history split-leakage proof is incomplete; "
    "no training rung may start."
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
    runtime = root / "engine/core/src/phase10r.rs"
    wasm = root / "engine/wasm/src/lib.rs"
    model_module = root / "training/open_shogi_training/phase10r_model.py"
    training_module = root / "training/open_shogi_training/phase10r_training.py"
    scanner_module = root / "training/open_shogi_training/data/phase10r_scan.py"
    runtime_present = runtime.is_file() and b"Osaval02Evaluator" in runtime.read_bytes()
    wasm_present = wasm.is_file() and b"WasmOsaval02Model" in wasm.read_bytes()
    exporter_present = model_module.is_file() and b"serialize_osaval02" in model_module.read_bytes()
    training_result: dict[str, Any]
    try:
        counts = {
            variant: sum(parameter.numel() for parameter in Phase10RModel(variant).parameters())
            for variant in (VARIANT_PAIR, VARIANT_PRIMARY)
        }
        expected_counts = {
            variant: expected_parameter_count(variant)
            for variant in (VARIANT_PAIR, VARIANT_PRIMARY)
        }
        example = Phase10RExample(
            sfen="lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
            source="aobazero",
            artifact_id="bounded-backend-fixture",
            record_id="bounded-backend-record",
            split="train",
            legal_moves=("7g7f", "2g2f"),
            played_move="7g7f",
            wdl=2,
            wdl_mask=True,
        )
        with tempfile.TemporaryDirectory(prefix="phase10r-backend-") as temporary:
            output_dir = Path(temporary) / "run"
            bounded = run_bounded_training(
                [example],
                output_dir=output_dir,
                manifest_sha256="1" * 64,
                config=TrainingConfig(
                    variant_id=VARIANT_PAIR,
                    requested_device="cpu",
                    max_steps=1,
                    batch_size=1,
                    checkpoint_interval_steps=1,
                    minimum_free_bytes=0,
                    enforce_disk=False,
                ),
            )
            model = Phase10RModel(VARIANT_PRIMARY, seed=7)
            artifact_path = Path(temporary) / "primary.osaval02"
            exported = export_osaval02_artifact(
                model,
                artifact_path,
                quantization="float32",
                dataset_manifest_sha256="1" * 64,
                training_run_reference="bounded-backend",
                git_commit="a" * 40,
            )
            parsed = parse_osaval02(artifact_path.read_bytes())
        training_result = {
            "status": "passed",
            "parameter_counts": counts,
            "expected_parameter_counts": expected_counts,
            "checkpoint": bounded["status"],
            "export": exported["status"],
            "parsed_variant": parsed.variant_id,
            "parsed_quantization": parsed.quantization,
        }
    except (OSError, RuntimeError, ValueError, Phase10RTrainingError) as error:
        training_result = {"status": "failed", "error": str(error)}
    passed = (
        runtime_present
        and wasm_present
        and exporter_present
        and training_module.is_file()
        and scanner_module.is_file()
        and training_result.get("status") == "passed"
    )
    return {
        "required_format": "OSAVAL02",
        "osaval02_parser_present": runtime_present,
        "osaval02_wasm_present": wasm_present,
        "phase10r_model_module_present": exporter_present,
        "phase10r_training_module_present": training_module.is_file(),
        "phase10r_scanner_module_present": scanner_module.is_file(),
        "bounded_training_validation": training_result,
        "available_legacy_format": "OSAVAL01",
        "pure_runtime_configured": True,
        "passed": passed,
        "stop_reason": (
            None if passed else "OSAVAL02 export/native/Wasm parity backend is incomplete."
        ),
    }


def _run_preflight(root: Path, argv: Sequence[str]) -> tuple[dict[str, Any], bool]:
    receipt = _base_receipt(root, "preflight", argv)
    failures: list[str] = []
    try:
        frozen = validate_phase10r(root)
        receipt["frozen_validation"] = "passed"
        receipt["frozen_hashes"] = _hash_manifest(root)
        receipt["mixture_control"] = frozen["canonical_pretraining_mixture"]
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
    try:
        leakage = scan_phase10r_population(root)
        receipt["split_leakage"] = leakage
        if leakage.get("status") != "passed":
            failures.append(LEAKAGE_GAP)
            failures.extend(
                "split leakage: "
                f"{failure.get('artifact_id', failure.get('source_id', 'unknown'))}: "
                f"{failure.get('reason', 'incomplete')}"
                for failure in leakage.get("failures", [])
            )
    except (OSError, ValueError, RuntimeError, Phase10RScanError) as error:
        receipt["split_leakage"] = {"status": "failed", "passed": False, "error": str(error)}
        failures.append(f"split leakage: {error}")
    try:
        preparation_path = preparation_manifest_path(root, "1m")
        if preparation_path.is_file() and not preparation_path.is_symlink():
            preparation = validate_preparation(root, "1m")
            receipt["preparation_control"] = {
                "status": "passed",
                "manifest": str(preparation_path.relative_to(root)),
                "manifest_sha256": preparation["manifest_sha256"],
                "source_stream_counts": preparation["source_stream_counts"],
                "source_statistics": preparation["source_statistics"],
                "deterministic_reproduction": preparation["deterministic_reproduction"],
                "leakage_validation": preparation["leakage_validation"],
                "legacy_preparation": preparation["legacy_preparation"],
            }
        else:
            receipt["preparation_control"] = {
                "status": "not_prepared",
                "manifest": str(preparation_path.relative_to(root)),
            }
    except (OSError, ValueError, RuntimeError, Phase10RExecutionError) as error:
        receipt["preparation_control"] = {"status": "failed", "error": str(error)}
        failures.append(f"preparation control: {error}")
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


def _prepare(root: Path, argv: Sequence[str], scale: str) -> int:
    receipt = _base_receipt(root, "prepare", argv)
    try:
        result = prepare_scale(root, scale)
        manifest = result.get("manifest")
        if not isinstance(manifest, Path):
            raise Phase10RRunError("preparation did not return a manifest path")
        receipt.update(
            {
                "status": "passed",
                "exit_status": 0,
                "scale": scale,
                "preparation": {
                    **result,
                    "manifest": str(manifest.relative_to(root)),
                },
            }
        )
    except (Phase10RExecutionError, OSError, RuntimeError, ValueError) as error:
        receipt.update(
            {
                "status": "blocked",
                "exit_status": 2,
                "scale": scale,
                "stop_reason": str(error),
            }
        )
    return _write_receipt_and_return(root, "prepare", argv, receipt)


def _receipt_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _receipt_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_receipt_value(item) for item in value]
    return value


def _campaign_operation(
    root: Path,
    command: str,
    argv: Sequence[str],
    operation: Any,
) -> int:
    receipt = _base_receipt(root, command, argv)
    try:
        result = operation()
        receipt.update(
            {
                "status": "passed",
                "exit_status": 0,
                "result": _receipt_value(result),
            }
        )
    except (Phase10RCampaignError, OSError, RuntimeError, ValueError) as error:
        receipt.update(
            {
                "status": "blocked",
                "exit_status": 2,
                "stop_reason": str(error),
            }
        )
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
    current_commit = receipt["git"].get("commit")
    campaign_receipts = [
        item
        for item in prior_receipts
        if item.get("command") != "report"
        and (
            current_commit is None
            or not isinstance(item.get("git"), dict)
            or item["git"].get("commit") == current_commit
        )
    ]
    stop_reasons: list[str] = []
    for item in campaign_receipts:
        failures = item.get("failures", [])
        if isinstance(failures, list):
            for reason in failures:
                if isinstance(reason, str) and reason not in stop_reasons:
                    stop_reasons.append(reason)
        reason = item.get("stop_reason")
        if isinstance(reason, str) and reason not in stop_reasons:
            stop_reasons.append(reason)

    def passed(command: str) -> list[dict[str, Any]]:
        return [
            item
            for item in campaign_receipts
            if item.get("command") == command and item.get("status") == "passed"
        ]

    preflight_receipts = passed("preflight")
    prepare_receipts = passed("prepare")
    train_receipts = passed("train")
    evaluate_receipts = passed("evaluate")
    selection_receipts = [
        item for item in campaign_receipts if item.get("command") == "select-hard"
    ]
    label_receipts = [item for item in campaign_receipts if item.get("command") == "label-hard"]
    trained_variants = [
        item.get("result") for item in train_receipts if isinstance(item.get("result"), dict)
    ]
    preparation = [
        item.get("preparation")
        for item in prepare_receipts
        if isinstance(item.get("preparation"), dict)
    ]
    evaluation = [
        item.get("result") for item in evaluate_receipts if isinstance(item.get("result"), dict)
    ]
    if not preflight_receipts:
        next_command = (
            "PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run "
            "preflight --root ."
        )
    elif not prepare_receipts:
        next_command = (
            "PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run "
            f"prepare --root . --scale {scale}"
        )
    else:
        trained_ids = {
            str(result.get("variant")) for result in trained_variants if isinstance(result, dict)
        }
        missing_variant = next(
            (variant for variant in TRAIN_VARIANTS if variant not in trained_ids), None
        )
        if missing_variant is not None:
            next_command = (
                "PYTHONPATH=training uv run --frozen python -m "
                f"open_shogi_training.phase10r_run train --root . --scale {scale} "
                f"--variant {missing_variant} --resume"
            )
        elif not evaluate_receipts:
            next_command = (
                "PYTHONPATH=training uv run --frozen python -m "
                "open_shogi_training.phase10r_run evaluate --root . "
                f"--scale {scale} --all-source-held-out --cross-runtime --incremental-parity"
            )
        elif not selection_receipts:
            next_command = (
                "PYTHONPATH=training uv run --frozen python -m "
                f"open_shogi_training.phase10r_run select-hard --root . --scale {scale}"
            )
        elif not label_receipts:
            next_command = (
                "PYTHONPATH=training uv run --frozen python -m "
                f"open_shogi_training.phase10r_run label-hard --root . --scale {scale} --resume"
            )
        else:
            next_command = (
                "A frozen post-label gate remains; inspect the completed rung receipt and stop "
                "for the mandatory review before any scale expansion."
            )
    blocked = any(item.get("status") == "blocked" for item in campaign_receipts)
    status = (
        "blocked"
        if blocked or stop_reasons
        else ("passed" if campaign_receipts else "no_execution_receipts")
    )
    report_receipts = [
        str(path.relative_to(root))
        for path in sorted(runs.glob("*.json"))
        if not path.name.startswith("PHASE10R_EXECUTION_REPORT")
    ]
    report_data = {
        "schema": "open_shogi_ai_phase10r_execution_report/v1",
        "status": status,
        "scale_requested": scale,
        "branch": receipt["git"]["branch"],
        "commit": receipt["git"]["commit"],
        "dirty": receipt["git"]["dirty"],
        "elapsed": "derived from immutable command receipts",
        "preflight": preflight_receipts,
        "preparation": preparation,
        "acquired_this_campaign": {"status": "not_performed"},
        "consumed_examples": {
            "status": "prepared_stream_only",
            "rungs": preparation,
        },
        "deduplicated_examples": {
            "status": "recorded_by_preparation_manifest",
            "rungs": preparation,
        },
        "teacher_labels_this_campaign": 0,
        "trained_variants": trained_variants,
        "evaluation": evaluation,
        "arena_results": [],
        "bootstrapped_generations": [],
        "pure_selfplay_generations": [],
        "current_best_pure_candidate": None,
        "sol_review_gate_reached": False,
        "final_objective_passed": False,
        "promotion": {"overall_champion_mutated": False, "promotion_performed": False},
        "stop_reasons": stop_reasons,
        "receipts": report_receipts,
        "next_exact_command": next_command,
    }
    json_report = runs / "PHASE10R_EXECUTION_REPORT.json"
    markdown_report = runs / "PHASE10R_EXECUTION_REPORT.md"
    if (
        json_report.exists()
        or json_report.is_symlink()
        or markdown_report.exists()
        or markdown_report.is_symlink()
    ):
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        json_report = runs / f"PHASE10R_EXECUTION_REPORT-{scale}-{stamp}.json"
        markdown_report = runs / f"PHASE10R_EXECUTION_REPORT-{scale}-{stamp}.md"
    _write_new_json(json_report, report_data)
    _write_text_new(
        markdown_report,
        "# Phase 10R-C execution report\n\n"
        f"Status: **{status}**\n\n"
        "This report is derived from immutable command receipts. Protected holdout content was "
        "not inspected.\n\n"
        "## Stop reasons\n\n"
        + ("\n".join(f"- {reason}" for reason in stop_reasons) or "- None recorded")
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


def _validate_teacher_binding(root: Path, scale: str) -> dict[str, Any]:
    control = load_teacher_binding_control(root)
    candidates = completed_teacher_bound_candidates(root, scale)
    if not candidates:
        raise Phase10RRunError("no completed teacher-bound candidate lineage exists")
    return {
        "schema": "open_shogiai_phase10r_teacher_binding_validation/v1",
        "status": "passed",
        "binding_version": control["binding_version"],
        "scale": scale,
        "candidates": candidates,
    }


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
    validate_binding = subparsers.add_parser("validate-teacher-binding")
    validate_binding.add_argument("--root", type=Path, default=Path("."))
    validate_binding.add_argument("--scale", type=_parse_scale, required=True)
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
    if command == "prepare":
        return _prepare(root, command_argv, args.scale)
    if command == "train":
        return _campaign_operation(
            root,
            command,
            command_argv,
            lambda: train_variant(
                root,
                args.scale,
                args.variant,
                resume=args.resume,
                git_commit=_git(root, "rev-parse", "HEAD"),
            ),
        )
    if command == "evaluate":
        return _campaign_operation(
            root,
            command,
            command_argv,
            lambda: evaluate_scale(
                root,
                args.scale,
                all_source_held_out=args.all_source_held_out,
                cross_runtime=args.cross_runtime,
                incremental_parity=args.incremental_parity,
                git_commit=_git(root, "rev-parse", "HEAD"),
            ),
        )
    if command == "select-hard":
        return _campaign_operation(
            root,
            command,
            command_argv,
            lambda: select_hard(root, args.scale, git_commit=_git(root, "rev-parse", "HEAD")),
        )
    if command == "label-hard":
        return _campaign_operation(
            root,
            command,
            command_argv,
            lambda: label_hard(
                root,
                args.scale,
                resume=args.resume,
                git_commit=_git(root, "rev-parse", "HEAD"),
            ),
        )
    if command == "validate-teacher-binding":
        return _campaign_operation(
            root,
            command,
            command_argv,
            lambda: _validate_teacher_binding(root, args.scale),
        )
    if command == "report":
        return _report(root, command_argv, args.scale)
    return _blocked_operation(root, command, command_argv, BACKEND_GAP)


if __name__ == "__main__":
    raise SystemExit(main())
