"""Measured node-budget and memory gate for staged teacher labeling."""

from __future__ import annotations

import hashlib
import math
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from open_shogi_training.labeling.artifacts import (
    ArtifactError,
    FileDigest,
    compact_json_bytes,
    load_json_object,
    write_json_atomic,
)
from open_shogi_training.labeling.config import TeacherConfig
from open_shogi_training.labeling.fingerprint import TeacherFingerprint, fingerprint_teacher
from open_shogi_training.labeling.selection import (
    SelectedPosition,
    SelectionResult,
    select_positions,
)
from open_shogi_training.labeling.usi import USIEngine, USIError, USIRetryError

BENCHMARK_SCHEMA: Final = "phase4_teacher_benchmark/v2"
LEGACY_BENCHMARK_SCHEMA: Final = "phase4_teacher_benchmark/v1"
MAX_BENCHMARK_BYTES: Final = 4 * 1024 * 1024
EngineFactory = Callable[[TeacherConfig, Path], USIEngine]


class BenchmarkError(ValueError):
    """Raised when a benchmark cannot establish a safe configured budget."""


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    path: Path
    digest: FileDigest
    selected_nodes: int | None
    report: dict[str, Any]


def run_benchmark(
    *,
    config: TeacherConfig,
    project_root: Path,
    positions_path: Path,
    dataset_manifest_path: Path,
    output_path: Path,
    engine_factory: EngineFactory = USIEngine,
) -> BenchmarkResult:
    """Measure configured node candidates and atomically publish a non-overwriting report."""

    selection = select_positions(positions_path, dataset_manifest_path, config.selection)
    fingerprint = fingerprint_teacher(config, project_root)
    teacher_identity_sha256 = _teacher_identity_sha256(fingerprint)
    positions = selection.positions[: config.benchmark.positions]
    if len(positions) < config.benchmark.positions:
        raise BenchmarkError(
            f"benchmark requires {config.benchmark.positions} positions, selected {len(positions)}"
        )
    engine = engine_factory(config, project_root)
    results: list[dict[str, Any]] = []
    reported_identity: dict[str, str | None] | None = None
    try:
        identity = engine.start()
        reported_identity = {"name": identity.name, "author": identity.author}
        for nodes in config.benchmark.node_candidates:
            searches: list[dict[str, Any]] = []
            for position in positions:
                sampler = _RssSampler(lambda: engine.pid, config.benchmark.rss_poll_ms)
                sampler.start()
                search_result = None
                search_error: USIError | None = None
                try:
                    search_result = engine.analyze_with_retry(position.canonical_sfen, nodes=nodes)
                except USIError as error:
                    search_error = error
                finally:
                    peak_rss = sampler.stop()
                if search_error is not None:
                    error_category = (
                        search_error.last_category
                        if isinstance(search_error, USIRetryError)
                        else search_error.category
                    )
                    searches.append(
                        {
                            "position_id": position.position_id,
                            "canonical_state_sha256": position.canonical_state_sha256,
                            "teacher_identity_sha256": teacher_identity_sha256,
                            "status": "failed",
                            "error_category": error_category,
                            "error_message": str(search_error)[:2_048],
                            "elapsed_ms": None,
                            "reported_nodes": None,
                            "depth": None,
                            "peak_rss_bytes": peak_rss,
                        }
                    )
                    continue
                if search_result is None:
                    raise AssertionError("teacher search ended without a result or error")
                if (
                    engine.identity is None
                    or {
                        "name": engine.identity.name,
                        "author": engine.identity.author,
                    }
                    != reported_identity
                ):
                    raise BenchmarkError("teacher USI identity changed during the benchmark")
                result = search_result
                searches.append(
                    {
                        "position_id": position.position_id,
                        "canonical_state_sha256": position.canonical_state_sha256,
                        "teacher_identity_sha256": teacher_identity_sha256,
                        "status": "completed",
                        "error_category": None,
                        "error_message": None,
                        "elapsed_ms": result.elapsed_ms,
                        "reported_nodes": result.primary.nodes,
                        "depth": result.primary.depth,
                        "peak_rss_bytes": peak_rss,
                    }
                )
            results.append(_summarize_budget(nodes, searches, config))
    finally:
        engine.close()
    selected_nodes = max(
        (entry["nodes"] for entry in results if entry["passes"]),
        default=None,
    )
    report: dict[str, Any] = {
        "schema": BENCHMARK_SCHEMA,
        "created_at": _utc_now(),
        "config_sha256": config.sha256,
        "dataset_manifest_sha256": selection.dataset_manifest_sha256,
        "positions_sha256": selection.positions_sha256,
        "selection_sha256": selection.selection_sha256,
        "sample_position_ids": [position.position_id for position in positions],
        "teacher": fingerprint.identity_record(),
        "teacher_identity_sha256": teacher_identity_sha256,
        "reported_identity": reported_identity,
        "resources": {
            "reference_host_memory_gib": config.reference_host_memory_gib,
            "working_memory_limit_gib": config.working_memory_limit_gib,
            "concurrency": config.concurrency,
            "threads": config.threads,
            "usi_hash_mb": config.hash_mb,
            "rss_measurement": "ps process-tree RSS summed during each search",
            "rss_poll_ms": config.benchmark.rss_poll_ms,
            "max_peak_rss_mib": config.benchmark.max_peak_rss_mib,
            "require_peak_rss": config.benchmark.require_peak_rss,
            "max_p95_ms": config.benchmark.max_p95_ms,
        },
        "budgets": results,
        "selected_nodes": selected_nodes,
        "configured_nodes": config.nodes,
        "labeling_authorized": selected_nodes == config.nodes,
    }
    digest = write_json_atomic(output_path, report, replace=False)
    return BenchmarkResult(output_path, digest, selected_nodes, report)


def verify_benchmark_report(
    path: Path,
    *,
    config: TeacherConfig,
    selection: SelectionResult,
    fingerprint: TeacherFingerprint,
) -> tuple[dict[str, Any], FileDigest]:
    """Recompute authorization exclusively from closed, fingerprint-bound raw rows."""

    try:
        report, digest = load_json_object(path, max_bytes=MAX_BENCHMARK_BYTES)
    except ArtifactError as error:
        raise BenchmarkError(str(error)) from error
    schema = report.get("schema")
    required = {
        "schema",
        "created_at",
        "config_sha256",
        "dataset_manifest_sha256",
        "positions_sha256",
        "selection_sha256",
        "sample_position_ids",
        "teacher",
        "reported_identity",
        "resources",
        "budgets",
        "selected_nodes",
        "configured_nodes",
        "labeling_authorized",
    }
    if schema == BENCHMARK_SCHEMA:
        required.add("teacher_identity_sha256")
    if schema not in {LEGACY_BENCHMARK_SCHEMA, BENCHMARK_SCHEMA} or set(report) != required:
        raise BenchmarkError("benchmark report schema or keys are invalid")
    _validate_timestamp(report["created_at"], "benchmark.created_at")
    if report["config_sha256"] != config.sha256:
        raise BenchmarkError("benchmark config hash does not match labeling config")
    if report["dataset_manifest_sha256"] != selection.dataset_manifest_sha256:
        raise BenchmarkError("benchmark dataset manifest hash does not match")
    if report["positions_sha256"] != selection.positions_sha256:
        raise BenchmarkError("benchmark positions hash does not match")
    expected_selection_sha256 = (
        selection.selection_sha256
        if schema == BENCHMARK_SCHEMA
        else selection.legacy_selection_sha256
    )
    if report["selection_sha256"] != expected_selection_sha256:
        raise BenchmarkError("benchmark selection hash does not match")
    expected_sample_ids = [
        position.position_id for position in selection.positions[: config.benchmark.positions]
    ]
    if report["sample_position_ids"] != expected_sample_ids:
        raise BenchmarkError("benchmark sample positions do not match deterministic selection")
    if not _strict_equal(report["teacher"], fingerprint.identity_record()):
        raise BenchmarkError("benchmark teacher files/options do not match")
    teacher_identity_sha256 = _teacher_identity_sha256(fingerprint)
    if schema == BENCHMARK_SCHEMA and report["teacher_identity_sha256"] != teacher_identity_sha256:
        raise BenchmarkError("benchmark teacher identity digest is invalid")
    _validate_reported_identity(report["reported_identity"])
    resources = report["resources"]
    if not isinstance(resources, dict) or resources.get("rss_measurement") not in {
        "ps process RSS polled during each search",
        "ps process-tree RSS summed during each search",
    }:
        raise BenchmarkError("benchmark RSS measurement method is invalid")
    expected_resources: dict[str, object] = {
        "reference_host_memory_gib": config.reference_host_memory_gib,
        "working_memory_limit_gib": config.working_memory_limit_gib,
        "concurrency": config.concurrency,
        "threads": config.threads,
        "usi_hash_mb": config.hash_mb,
        "rss_measurement": (
            resources.get("rss_measurement") if isinstance(resources, dict) else None
        ),
        "rss_poll_ms": config.benchmark.rss_poll_ms,
        "max_peak_rss_mib": config.benchmark.max_peak_rss_mib,
        "require_peak_rss": config.benchmark.require_peak_rss,
        "max_p95_ms": config.benchmark.max_p95_ms,
    }
    if not _strict_equal(resources, expected_resources):
        raise BenchmarkError("benchmark resource settings do not match")
    budgets = report["budgets"]
    if not isinstance(budgets, list) or len(budgets) != len(config.benchmark.node_candidates):
        raise BenchmarkError("benchmark budget results are missing")
    selected_by_id = {position.position_id: position for position in selection.positions}
    for index, (budget, expected_nodes) in enumerate(
        zip(budgets, config.benchmark.node_candidates, strict=True)
    ):
        if not isinstance(budget, dict) or set(budget) != _BUDGET_KEYS:
            raise BenchmarkError(f"benchmark budget {index} keys are invalid")
        if budget["nodes"] != expected_nodes:
            raise BenchmarkError(f"benchmark budget {index} identity is invalid")
        searches = budget["searches"]
        if not isinstance(searches, list) or len(searches) != len(expected_sample_ids):
            raise BenchmarkError(f"benchmark budget {index} searches are invalid")
        for row_index, (search, expected_position_id) in enumerate(
            zip(searches, expected_sample_ids, strict=True)
        ):
            _validate_search_row(
                search,
                schema=schema,
                expected_position=selected_by_id[expected_position_id],
                expected_teacher_sha256=teacher_identity_sha256,
                expected_nodes=expected_nodes,
                budget_index=index,
                row_index=row_index,
            )
        recomputed = _summarize_budget(expected_nodes, searches, config)
        if not _strict_equal(budget, recomputed):
            raise BenchmarkError(
                f"benchmark budget {index} summaries or gate decision do not match raw rows"
            )
    measured_selection = max(
        (budget["nodes"] for budget in budgets if budget["passes"]),
        default=None,
    )
    expected_authorized = measured_selection == config.nodes
    if (
        not _strict_equal(report["selected_nodes"], measured_selection)
        or not _strict_equal(report["configured_nodes"], config.nodes)
        or report["labeling_authorized"] is not expected_authorized
        or not expected_authorized
    ):
        raise BenchmarkError(
            "benchmark did not select the configured node budget; update config and rerun"
        )
    return report, digest


_BUDGET_KEYS: Final = {
    "nodes",
    "searches",
    "completed",
    "failed",
    "elapsed_ms",
    "peak_rss_bytes",
    "memory_measured_for_all",
    "passes",
}
_LEGACY_SEARCH_KEYS: Final = {
    "position_id",
    "status",
    "error_category",
    "error_message",
    "elapsed_ms",
    "reported_nodes",
    "depth",
    "peak_rss_bytes",
}
_SEARCH_KEYS: Final = _LEGACY_SEARCH_KEYS | {
    "canonical_state_sha256",
    "teacher_identity_sha256",
}


def _validate_search_row(
    value: object,
    *,
    schema: object,
    expected_position: SelectedPosition,
    expected_teacher_sha256: str,
    expected_nodes: int,
    budget_index: int,
    row_index: int,
) -> None:
    prefix = f"benchmark budget {budget_index} search {row_index}"
    keys = _SEARCH_KEYS if schema == BENCHMARK_SCHEMA else _LEGACY_SEARCH_KEYS
    if not isinstance(value, dict) or set(value) != keys:
        raise BenchmarkError(f"{prefix} keys are invalid")
    if value["position_id"] != expected_position.position_id:
        raise BenchmarkError(f"{prefix} position identity is invalid")
    if schema == BENCHMARK_SCHEMA and (
        value["canonical_state_sha256"] != expected_position.canonical_state_sha256
        or value["teacher_identity_sha256"] != expected_teacher_sha256
    ):
        raise BenchmarkError(f"{prefix} selection/teacher binding is invalid")
    status = value["status"]
    if status == "completed":
        peak = value["peak_rss_bytes"]
        if peak is not None:
            _require_nonnegative_int(peak, f"{prefix}.peak_rss_bytes", positive=True)
        if value["error_category"] is not None or value["error_message"] is not None:
            raise BenchmarkError(f"{prefix} completed row contains an error")
        _require_nonnegative_int(value["elapsed_ms"], f"{prefix}.elapsed_ms")
        reported_nodes = _require_nonnegative_int(
            value["reported_nodes"],
            f"{prefix}.reported_nodes",
            positive=True,
        )
        if reported_nodes < expected_nodes:
            raise BenchmarkError(f"{prefix}.reported_nodes is below the requested budget")
        _require_nonnegative_int(value["depth"], f"{prefix}.depth")
        return
    if status != "failed":
        raise BenchmarkError(f"{prefix} status is invalid")
    peak = value["peak_rss_bytes"]
    if peak is not None:
        _require_nonnegative_int(peak, f"{prefix}.peak_rss_bytes")
    if value["error_category"] not in {"process", "protocol", "timeout"}:
        raise BenchmarkError(f"{prefix} failure category is invalid")
    message = value["error_message"]
    if (
        not isinstance(message, str)
        or not message
        or len(message) > 2_048
        or any(character in message for character in "\0")
    ):
        raise BenchmarkError(f"{prefix} failure message is invalid")
    if any(value[field] is not None for field in ("elapsed_ms", "reported_nodes", "depth")):
        raise BenchmarkError(f"{prefix} failed row contains completed measurements")


def _summarize_budget(
    nodes: int,
    searches: list[dict[str, Any]],
    config: TeacherConfig,
) -> dict[str, Any]:
    elapsed_values = [
        search["elapsed_ms"] for search in searches if search["status"] == "completed"
    ]
    peak_values = [
        search["peak_rss_bytes"] for search in searches if search["peak_rss_bytes"] is not None
    ]
    completed = len(elapsed_values)
    failed = len(searches) - completed
    p95_ms = _nearest_rank_percentile(elapsed_values, 0.95)
    peak_rss_bytes = max(peak_values) if peak_values else None
    memory_measured = len(peak_values) == len(searches)
    passes = (
        failed == 0
        and completed == len(searches)
        and p95_ms is not None
        and p95_ms <= config.benchmark.max_p95_ms
        and (
            not config.benchmark.require_peak_rss
            or (
                memory_measured
                and peak_rss_bytes is not None
                and peak_rss_bytes <= config.benchmark.max_peak_rss_mib * 1024 * 1024
            )
        )
    )
    return {
        "nodes": nodes,
        "searches": searches,
        "completed": completed,
        "failed": failed,
        "elapsed_ms": {
            "min": min(elapsed_values) if elapsed_values else None,
            "median": _nearest_rank_percentile(elapsed_values, 0.5),
            "p95": p95_ms,
            "max": max(elapsed_values) if elapsed_values else None,
        },
        "peak_rss_bytes": peak_rss_bytes,
        "memory_measured_for_all": memory_measured,
        "passes": passes,
    }


def _teacher_identity_sha256(fingerprint: TeacherFingerprint) -> str:
    return hashlib.sha256(compact_json_bytes(fingerprint.identity_record())).hexdigest()


def _validate_reported_identity(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {"name", "author"}:
        raise BenchmarkError("benchmark reported teacher identity is invalid")
    name = value["name"]
    author = value["author"]
    if not isinstance(name, str) or not name or len(name) > 256:
        raise BenchmarkError("benchmark reported teacher name is invalid")
    if author is not None and (not isinstance(author, str) or not author or len(author) > 256):
        raise BenchmarkError("benchmark reported teacher author is invalid")
    if any(character in name for character in "\r\n\0") or (
        isinstance(author, str) and any(character in author for character in "\r\n\0")
    ):
        raise BenchmarkError("benchmark reported teacher identity contains a delimiter")


def _validate_timestamp(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.endswith("Z") or len(value) > 64:
        raise BenchmarkError(f"{name} is invalid")
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError as error:
        raise BenchmarkError(f"{name} is invalid") from error
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise BenchmarkError(f"{name} is not UTC")


def _require_nonnegative_int(value: object, name: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BenchmarkError(f"{name} must be an integer at least {minimum}")
    return value


def _strict_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_strict_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _strict_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right


class _RssSampler:
    def __init__(self, pid: Callable[[], int | None], poll_ms: int) -> None:
        self._pid = pid
        self._poll_seconds = poll_ms / 1_000
        self._stop = threading.Event()
        self._peak: int | None = None
        self._thread = threading.Thread(target=self._run, name="teacher-rss", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> int | None:
        self._stop.set()
        self._thread.join(max(1.0, self._poll_seconds * 2))
        return self._peak

    def _run(self) -> None:
        while not self._stop.is_set():
            pid = self._pid()
            if pid is not None:
                measured = _read_process_tree_rss(pid)
                if measured is not None and (self._peak is None or measured > self._peak):
                    self._peak = measured
            self._stop.wait(self._poll_seconds)
        pid = self._pid()
        if pid is not None:
            measured = _read_process_tree_rss(pid)
            if measured is not None and (self._peak is None or measured > self._peak):
                self._peak = measured


def _read_process_tree_rss(pid: int) -> int | None:
    try:
        completed = subprocess.run(
            ["/bin/ps", "-axo", "pid=,ppid=,pgid=,rss="],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=1,
            env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "LC_ALL": "C", "LANG": "C"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0 or len(completed.stdout) > 8 * 1024 * 1024:
        return None
    processes: dict[int, tuple[int, int, int]] = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) != 4:
            return None
        try:
            process_id, parent_id, group_id, rss_kib = (int(field) for field in fields)
        except ValueError:
            return None
        if (
            process_id <= 0
            or parent_id < 0
            or group_id < 0
            or rss_kib < 0
            or process_id in processes
        ):
            return None
        processes[process_id] = (parent_id, group_id, rss_kib)
    selected = {process_id for process_id, (_, group_id, _) in processes.items() if group_id == pid}
    if pid in processes:
        selected.add(pid)
    changed = True
    while changed:
        changed = False
        for process_id, (parent_id, _, _) in processes.items():
            if parent_id in selected and process_id not in selected:
                selected.add(process_id)
                changed = True
    if not selected:
        return None
    return sum(processes[process_id][2] for process_id in selected) * 1024


def _nearest_rank_percentile(values: list[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
