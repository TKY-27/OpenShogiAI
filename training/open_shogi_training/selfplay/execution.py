"""Bounded no-shell command execution and resumable paired-job state."""

from __future__ import annotations

import hashlib
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final

from open_shogi_training.labeling.artifacts import stable_directory_lock
from open_shogi_training.labeling.execution import (
    ExecutableSnapshot,
    ExecutableSnapshotError,
    RuntimeTreeSnapshot,
    RuntimeTreeSnapshotError,
)
from open_shogi_training.labeling.process_identity import read_process_identity

from .common import (
    ArtifactRef,
    ContractError,
    artifact_ref,
    canonical_sha256,
    contained_path,
    ensure_contained_directory,
    load_bytes_artifact,
    load_json,
    load_json_artifact,
    replace_json_state,
    require_bool,
    require_clean_head,
    require_enum,
    require_exact_keys,
    require_identifier,
    require_int,
    require_list,
    require_mapping,
    require_relative_path,
    require_sha256,
    require_string,
    validate_relative_path,
    validate_utc_timestamp,
    verify_artifact_ref,
    write_bytes_new,
    write_json_new,
)
from .config import (
    MAX_WORKING_MEMORY_MIB,
    PHASE6_GAMES,
    PHASE6_MAX_PLIES,
    PHASE6_NODES_PER_MOVE,
    PHASE6_NORMAL_START_PAIRS,
    PHASE6_SEED,
    PHASE6_START_SET_PAIRS,
    PHASE6_WORKERS,
)
from .planning import ARENA_PLAN_SCHEMA, INITIAL_SFEN, SELFPLAY_PLAN_SCHEMA
from .runtime_receipt import (
    PythonRuntimeAuthority,
    validate_python_runtime_receipt,
)

JOB_STATE_SCHEMA: Final = "phase6_paired_job_state/v1"
SELFPLAY_MANIFEST_SCHEMA: Final = "phase6_selfplay_manifest/v1"
ARENA_EXECUTION_MANIFEST_SCHEMA: Final = "phase6_arena_execution_manifest/v1"
MAX_COMMAND_ARGUMENTS: Final = 256
MAX_ARGUMENT_BYTES: Final = 16 * 1024
MAX_COMMAND_OUTPUT_BYTES: Final = 4 * 1024 * 1024
_PROCESS_IDENTITY_BIND_TIMEOUT_SECONDS: Final = 1.0
_PROCESS_IDENTITY_BIND_RETRY_SECONDS: Final = 0.005

_PLAN_KEYS = {
    SELFPLAY_PLAN_SCHEMA: frozenset(
        {
            "schema",
            "generationId",
            "champion",
            "engine",
            "engineBuildReceipt",
            "modelRegistry",
            "gitCommit",
            "config",
            "configSha256",
            "startPositions",
            "startPositionValidation",
            "datasetManifest",
            "gameCount",
            "pairCount",
            "normalStartPairs",
            "startSetPairs",
            "seed",
            "nodesPerMove",
            "maxWorkers",
            "memoryLimitMiB",
            "memoryPerWorkerMiB",
            "jobs",
            "planSha256",
        }
    ),
    ARENA_PLAN_SCHEMA: frozenset(
        {
            "schema",
            "generationId",
            "champion",
            "challenger",
            "engine",
            "engineBuildReceipt",
            "modelRegistry",
            "gitCommit",
            "config",
            "configSha256",
            "startPositions",
            "startPositionValidation",
            "datasetManifest",
            "gameCount",
            "pairCount",
            "normalStartPairs",
            "startSetPairs",
            "seed",
            "nodesPerMove",
            "maxWorkers",
            "memoryLimitMiB",
            "memoryPerWorkerMiB",
            "jobs",
            "planSha256",
        }
    ),
}
_JOB_KEYS = frozenset(
    {
        "jobId",
        "pairIndex",
        "startGroup",
        "startPositionId",
        "sfen",
        "seed",
        "gameIds",
        "modelAColorOrder",
        "outputDir",
        "reportPath",
        "csaPaths",
        "quarantinePath",
        "command",
    }
)
_COMMAND_KEYS = frozenset({"kind", "argv", "timeoutSeconds"})
_STATE_KEYS = frozenset({"schema", "revision", "planSha256", "status", "attempts"})
_ATTEMPT_KEYS = frozenset(
    {
        "jobId",
        "attempt",
        "status",
        "returnCode",
        "timedOut",
        "outputLimitExceeded",
        "memoryLimitExceeded",
        "peakRssBytes",
        "rssMeasurement",
        "stdout",
        "stderr",
        "report",
        "csa",
        "quarantine",
        "failureCategory",
        "completedAt",
        "commandReceipt",
    }
)

_PROCESS_RECEIPT_SCHEMA: Final = "phase6_command_receipt/v2"
_PROCESS_RECEIPT_KEYS: Final = frozenset(
    {
        "schema",
        "command",
        "commandSha256",
        "resume",
        "memoryLimitMiB",
        "expectedExecutable",
        "engineBuildReceipt",
        "runtimeReceipt",
        "returnCode",
        "timedOut",
        "outputLimitExceeded",
        "memoryLimitExceeded",
        "peakRssBytes",
        "rssMeasurement",
        "stdout",
        "stderr",
    }
)
_ATTEMPT_RECEIPT_SCHEMA: Final = "phase6_attempt_command_receipt/v1"
_ATTEMPT_RECEIPT_KEYS: Final = frozenset(
    {
        "schema",
        "planSha256",
        "jobId",
        "attempt",
        "command",
        "engine",
        "engineBuildReceipt",
        "processReceipt",
        "result",
        "stdout",
        "stderr",
        "report",
        "csa",
        "quarantine",
        "failureCategory",
        "completedAt",
        "receiptSha256",
    }
)
_ATTEMPT_RESULT_KEYS: Final = frozenset(
    {
        "returnCode",
        "timedOut",
        "outputLimitExceeded",
        "memoryLimitExceeded",
        "peakRssBytes",
        "rssMeasurement",
    }
)


@dataclass(frozen=True, slots=True)
class CommandOutcome:
    return_code: int
    timed_out: bool
    output_limit_exceeded: bool
    memory_limit_exceeded: bool
    peak_rss_bytes: int | None
    rss_measurement: str
    stdout: ArtifactRef
    stderr: ArtifactRef
    command_receipt: ArtifactRef | None = None
    runtime_receipt: ArtifactRef | None = None
    engine_build_receipt: ArtifactRef | None = None


@dataclass(slots=True)
class _PythonSourceSnapshot:
    runtime: RuntimeTreeSnapshot
    identity: dict[str, object]

    @property
    def cwd(self) -> Path:
        return self.runtime.cwd

    def seal(self) -> None:
        self.runtime.seal()

    def assert_unchanged(self) -> None:
        self.runtime.assert_unchanged()

    def close(self) -> None:
        self.runtime.close()


class _ObservedProcessIds(set[int]):
    """PID set carrying the sampled process-start identity for safe later signals."""

    def __init__(self, values: set[int], identities: Mapping[int, str]) -> None:
        super().__init__(values)
        self.identities = dict(identities)


def _close_command_resources(
    snapshot: ExecutableSnapshot | None,
    python_authority: PythonRuntimeAuthority | None,
    python_source: _PythonSourceSnapshot | None,
) -> None:
    """Close every launch authority and preserve the active/first failure."""

    errors: list[BaseException] = []
    for resource in (snapshot, python_authority, python_source):
        if resource is None:
            continue
        try:
            resource.close()
        except BaseException as error:
            errors.append(error)
    if not errors:
        return
    active = sys.exception()
    if active is not None:
        for error in errors:
            active.add_note(f"command runtime cleanup also failed: {error!r}")
        return
    first = errors[0]
    for additional in errors[1:]:
        first.add_note(f"additional command runtime cleanup failure: {additional!r}")
    raise first


class CommandRunner:
    """Execute only the repository engine CLI or the exact model Python module."""

    def __init__(
        self,
        repository_root: Path,
        *,
        require_clean_repository: bool = False,
        require_engine_build_receipt: bool | None = None,
    ) -> None:
        self.repository_root = repository_root.resolve(strict=True)
        self.require_clean_repository = require_clean_repository
        self.require_engine_build_receipt = (
            require_clean_repository
            if require_engine_build_receipt is None
            else require_engine_build_receipt
        )

    def run(
        self,
        command: object,
        *,
        stdout_path: str,
        stderr_path: str,
        resume: bool = False,
        expected_executable: ArtifactRef | None = None,
        engine_build_receipt: ArtifactRef | None = None,
        memory_limit_mib: int = 4_096,
        receipt_path: str | None = None,
    ) -> CommandOutcome:
        table = require_mapping(command, "command")
        require_exact_keys(table, _COMMAND_KEYS, "command")
        kind = require_enum(
            table,
            "kind",
            "command",
            {
                "engine_arena",
                "engine_validation",
                "engine_dataset_export",
                "model_cli",
                "labeling_cli",
            },
        )
        timeout = require_int(table, "timeoutSeconds", "command", minimum=1, maximum=172_800)
        if (
            isinstance(memory_limit_mib, bool)
            or not isinstance(memory_limit_mib, int)
            or not 128 <= memory_limit_mib <= 8_192
        ):
            raise ContractError("command memory limit must be in 128..8192 MiB")
        argv = _command_argv(table, kind)
        supplied_commit: str | None = None
        clean_commit: str | None = None
        if self.require_clean_repository:
            if "--git-commit" in argv:
                index = argv.index("--git-commit")
                if index + 1 >= len(argv):
                    raise ContractError("git commit option has no value")
                supplied_commit = argv[index + 1]
            clean_commit = require_clean_head(self.repository_root, supplied_commit)
        if resume:
            if kind != "engine_arena":
                raise ContractError("resume is supported only for engine arena commands")
            argv.append("--resume")
        effective = self._materialize(kind, argv)
        expected_source: Path | None = None
        snapshot: ExecutableSnapshot | None = None
        python_source: _PythonSourceSnapshot | None = None
        python_authority: PythonRuntimeAuthority | None = None
        try:
            is_engine = kind in {
                "engine_arena",
                "engine_validation",
                "engine_dataset_export",
            }
            if is_engine:
                if expected_executable is None or expected_executable.path != argv[0]:
                    raise ContractError("engine execution requires its exact planned artifact")
                verify_artifact_ref(self.repository_root, expected_executable)
                expected_source = contained_path(
                    self.repository_root, expected_executable.path, must_exist=True
                )
                if self.require_engine_build_receipt and engine_build_receipt is None:
                    raise ContractError("engine execution requires its exact planned build receipt")
                if engine_build_receipt is not None:
                    from .engine_receipt import validate_engine_build_receipt

                    validate_engine_build_receipt(
                        self.repository_root,
                        engine_build_receipt,
                        expected_engine=expected_executable,
                        expected_git_commit=supplied_commit,
                    )
            else:
                if engine_build_receipt is not None:
                    raise ContractError("Python commands must not carry an engine build receipt")
                if not self.require_clean_repository or clean_commit is None:
                    raise ContractError(
                        "Python pipeline commands require a clean repository runtime snapshot"
                    )
                python_source = _snapshot_python_runtime(
                    self.repository_root,
                    supplied_commit=clean_commit,
                )
                python_authority = PythonRuntimeAuthority.create(
                    self.repository_root,
                    git_commit=clean_commit,
                    source_snapshot=python_source.identity,
                )
                python_source.seal()
                effective = _isolated_python_command(
                    python_authority.interpreter.executable_path,
                    python_source.cwd,
                    python_authority.site_paths,
                    argv,
                )

            runtime_receipt = None if python_authority is None else python_authority.receipt
            invocation = {
                "command": dict(table),
                "resume": resume,
                "memoryLimitMiB": memory_limit_mib,
                "expectedExecutable": (
                    None if expected_executable is None else expected_executable.as_dict()
                ),
                "engineBuildReceipt": (
                    None if engine_build_receipt is None else engine_build_receipt.as_dict()
                ),
                "runtimeReceipt": (None if runtime_receipt is None else runtime_receipt.as_dict()),
            }
            command_identity = canonical_sha256(invocation)
            if receipt_path is not None:
                receipt_path = validate_relative_path(receipt_path)
                receipt_file = contained_path(self.repository_root, receipt_path)
                if receipt_file.exists():
                    return validate_command_receipt(
                        self.repository_root,
                        artifact_ref(self.repository_root, receipt_path),
                        command=dict(table),
                        command_identity=command_identity,
                        resume=resume,
                        memory_limit_mib=memory_limit_mib,
                        expected_executable=expected_executable,
                        engine_build_receipt=engine_build_receipt,
                        runtime_receipt=runtime_receipt,
                        expected_git_commit=clean_commit,
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                    )
            stdout_path = validate_relative_path(stdout_path)
            stderr_path = validate_relative_path(stderr_path)
            stdout_file = contained_path(self.repository_root, stdout_path)
            stderr_file = contained_path(self.repository_root, stderr_path)
            if is_engine:
                assert expected_executable is not None and expected_source is not None
                snapshot_root = ensure_contained_directory(
                    self.repository_root, "local/runtime-snapshots"
                )
                snapshot = ExecutableSnapshot.create(
                    expected_source,
                    temporary_directory=snapshot_root,
                    max_bytes=512 * 1024 * 1024,
                    expected_sha256=expected_executable.sha256,
                )
                effective[0] = snapshot.executable_path

            def launch_guard() -> None:
                if snapshot is not None:
                    snapshot.assert_snapshot_unchanged()
                    snapshot.assert_source_unchanged()
                if python_source is not None and python_authority is not None:
                    python_source.assert_unchanged()
                    python_authority.assert_unchanged()
                    require_clean_head(self.repository_root, clean_commit)

            process_environment = _sanitized_environment()
            if python_authority is not None:
                process_environment.update(
                    {
                        "PYTHONHOME": python_authority.base_prefix,
                        "PYTHONDONTWRITEBYTECODE": "1",
                    }
                )
            (
                stdout_data,
                stderr_data,
                return_code,
                timed_out,
                output_limit,
                memory_limit_exceeded,
                peak_rss_bytes,
                rss_measurement,
            ) = _run_bounded_process(
                effective,
                cwd=self.repository_root,
                timeout_seconds=timeout,
                pass_fds=snapshot.pass_fds() if snapshot is not None else (),
                memory_limit_bytes=memory_limit_mib * 1024 * 1024,
                environment=process_environment,
                launch_guard=launch_guard,
            )
            launch_guard()
            _validate_rss_result(
                measurement=rss_measurement,
                peak_rss_bytes=peak_rss_bytes,
                memory_limit_bytes=memory_limit_mib * 1024 * 1024,
                memory_limit_exceeded=memory_limit_exceeded,
                completed=(
                    return_code == 0
                    and not timed_out
                    and not output_limit
                    and not memory_limit_exceeded
                ),
                context="bounded command result",
            )
            resources = (snapshot, python_authority, python_source)
            snapshot = None
            python_authority = None
            python_source = None
            _close_command_resources(*resources)
            stdout_hash, stdout_size = write_bytes_new(stdout_file, stdout_data)
            stderr_hash, stderr_size = write_bytes_new(stderr_file, stderr_data)
            stdout_ref = ArtifactRef(
                path=stdout_path,
                sha256=stdout_hash,
                size=stdout_size,
            )
            stderr_ref = ArtifactRef(
                path=stderr_path,
                sha256=stderr_hash,
                size=stderr_size,
            )
            receipt_ref: ArtifactRef | None = None
            if receipt_path is not None:
                write_json_new(
                    contained_path(self.repository_root, receipt_path),
                    {
                        "schema": _PROCESS_RECEIPT_SCHEMA,
                        **invocation,
                        "commandSha256": command_identity,
                        "returnCode": return_code,
                        "timedOut": timed_out,
                        "outputLimitExceeded": output_limit,
                        "memoryLimitExceeded": memory_limit_exceeded,
                        "peakRssBytes": peak_rss_bytes,
                        "rssMeasurement": rss_measurement,
                        "stdout": stdout_ref.as_dict(),
                        "stderr": stderr_ref.as_dict(),
                    },
                )
                receipt_ref = artifact_ref(self.repository_root, receipt_path)
            return CommandOutcome(
                return_code=return_code,
                timed_out=timed_out,
                output_limit_exceeded=output_limit,
                memory_limit_exceeded=memory_limit_exceeded,
                peak_rss_bytes=peak_rss_bytes,
                rss_measurement=rss_measurement,
                stdout=stdout_ref,
                stderr=stderr_ref,
                command_receipt=receipt_ref,
                runtime_receipt=runtime_receipt,
                engine_build_receipt=engine_build_receipt,
            )
        except (ExecutableSnapshotError, RuntimeTreeSnapshotError) as error:
            raise ContractError(f"runtime snapshot failed: {error}") from error
        finally:
            _close_command_resources(snapshot, python_authority, python_source)

    def _materialize(self, kind: str, argv: list[str]) -> list[str]:
        if kind in {"engine_arena", "engine_validation", "engine_dataset_export"}:
            executable = contained_path(self.repository_root, argv[0], must_exist=True)
            if executable.name != "open-shogi-cli" or not os.access(executable, os.X_OK):
                raise ContractError("engine command must use the executable repository CLI")
            required_subcommand = {
                "engine_arena": "arena",
                "engine_validation": "perft",
                "engine_dataset_export": "export-csa-jsonl",
            }[kind]
            if len(argv) < 2 or argv[1] != required_subcommand:
                raise ContractError(f"{kind} must use the {required_subcommand} subcommand")
            if kind == "engine_arena":
                _validate_runtime_arena_command(argv)
                _validate_contained_command_paths(
                    self.repository_root,
                    argv,
                    input_files={"--a-model", "--b-model"},
                    output_paths={"--output-dir"},
                )
            elif kind == "engine_validation":
                _validate_engine_validation_command(argv)
            else:
                _validate_dataset_export_command(argv)
                _validate_contained_command_paths(
                    self.repository_root,
                    argv,
                    input_directories={"--input-dir"},
                    output_paths={"--output"},
                )
            return [str(executable), *argv[1:]]
        if kind == "model_cli":
            _validate_model_train_command(argv)
            _validate_contained_command_paths(
                self.repository_root,
                argv,
                input_files={
                    "--features",
                    "--model",
                    "--training",
                    "--labels",
                    "--positions",
                    "--dataset-manifest",
                    "--replay-manifest",
                    "--resume",
                },
                output_paths={"--output-dir"},
            )
        else:
            _validate_labeling_command(argv)
            _validate_contained_command_paths(
                self.repository_root,
                argv,
                input_files={
                    "--config",
                    "--positions",
                    "--dataset-manifest",
                    "--benchmark-report",
                },
                output_paths={"--output-dir"},
            )
        return [sys.executable, *argv[1:]]


def validate_command_receipt(
    repository_root: Path,
    receipt_ref: ArtifactRef,
    *,
    command: dict[str, object],
    command_identity: str,
    resume: bool,
    memory_limit_mib: int,
    expected_executable: ArtifactRef | None,
    engine_build_receipt: ArtifactRef | None,
    runtime_receipt: ArtifactRef | None,
    expected_git_commit: str | None,
    stdout_path: str,
    stderr_path: str,
) -> CommandOutcome:
    """Revalidate one durable bounded-command receipt against its exact invocation."""

    root = require_mapping(load_json_artifact(repository_root, receipt_ref), "command receipt")
    require_exact_keys(root, _PROCESS_RECEIPT_KEYS, "command receipt")
    if root.get("schema") != _PROCESS_RECEIPT_SCHEMA:
        raise ContractError("unsupported command receipt schema")
    recorded_command = require_mapping(root.get("command"), "command receipt.command")
    require_exact_keys(recorded_command, _COMMAND_KEYS, "command receipt.command")
    if recorded_command != command:
        raise ContractError("command receipt records a different command")
    if require_bool(root, "resume", "command receipt") != resume:
        raise ContractError("command receipt records a different resume decision")
    if (
        require_int(
            root,
            "memoryLimitMiB",
            "command receipt",
            minimum=128,
            maximum=8_192,
        )
        != memory_limit_mib
    ):
        raise ContractError("command receipt records a different memory limit")
    recorded_executable = _optional_artifact_ref(
        root.get("expectedExecutable"), "command receipt.expectedExecutable"
    )
    recorded_engine = _optional_artifact_ref(
        root.get("engineBuildReceipt"), "command receipt.engineBuildReceipt"
    )
    recorded_runtime = _optional_artifact_ref(
        root.get("runtimeReceipt"), "command receipt.runtimeReceipt"
    )
    if recorded_executable != expected_executable:
        raise ContractError("command receipt records a different executable")
    if recorded_engine != engine_build_receipt:
        raise ContractError("command receipt records a different engine receipt")
    if recorded_runtime != runtime_receipt:
        raise ContractError("command receipt records a different Python runtime")
    if recorded_executable is None:
        if recorded_engine is not None or recorded_runtime is None or expected_git_commit is None:
            raise ContractError("Python command receipt has inconsistent runtime authorities")
        validate_python_runtime_receipt(
            repository_root,
            recorded_runtime,
            expected_git_commit=expected_git_commit,
        )
    elif recorded_runtime is not None:
        raise ContractError("engine command receipt must not carry a Python runtime")
    if require_sha256(root, "commandSha256", "command receipt") != command_identity:
        raise ContractError("command receipt belongs to a different execution")
    stdout = ArtifactRef.from_dict(root.get("stdout"), "command receipt.stdout")
    stderr = ArtifactRef.from_dict(root.get("stderr"), "command receipt.stderr")
    if stdout.path != stdout_path or stderr.path != stderr_path:
        raise ContractError("command receipt log paths differ from the requested attempt")
    verify_artifact_ref(repository_root, stdout)
    verify_artifact_ref(repository_root, stderr)
    peak = root.get("peakRssBytes")
    if peak is not None:
        peak = require_int(root, "peakRssBytes", "command receipt", minimum=0, maximum=2**63 - 1)
    measurement = require_enum(
        root,
        "rssMeasurement",
        "command receipt",
        {
            "process_tree_ps_rss_sum",
            "process_tree_ps_short_lived_no_sample",
            "unavailable",
        },
    )
    return_code = require_int(root, "returnCode", "command receipt", minimum=-255, maximum=255)
    timed_out = require_bool(root, "timedOut", "command receipt")
    output_limit = require_bool(root, "outputLimitExceeded", "command receipt")
    memory_exceeded = require_bool(root, "memoryLimitExceeded", "command receipt")
    _validate_rss_result(
        measurement=measurement,
        peak_rss_bytes=peak,
        memory_limit_bytes=memory_limit_mib * 1024 * 1024,
        memory_limit_exceeded=memory_exceeded,
        completed=(return_code == 0 and not timed_out and not output_limit and not memory_exceeded),
        context="command receipt",
    )
    return CommandOutcome(
        return_code=return_code,
        timed_out=timed_out,
        output_limit_exceeded=output_limit,
        memory_limit_exceeded=memory_exceeded,
        peak_rss_bytes=peak,
        rss_measurement=measurement,
        stdout=stdout,
        stderr=stderr,
        command_receipt=receipt_ref,
        runtime_receipt=recorded_runtime,
        engine_build_receipt=recorded_engine,
    )


def _optional_artifact_ref(value: object, context: str) -> ArtifactRef | None:
    return None if value is None else ArtifactRef.from_dict(value, context)


def _validate_rss_result(
    *,
    measurement: str,
    peak_rss_bytes: int | None,
    memory_limit_bytes: int,
    memory_limit_exceeded: bool,
    completed: bool,
    context: str,
) -> None:
    if measurement == "process_tree_ps_rss_sum":
        if peak_rss_bytes is None:
            raise ContractError(f"{context} RSS sum is missing its peak measurement")
        if completed and peak_rss_bytes > memory_limit_bytes:
            raise ContractError(f"{context} completed above its worker memory limit")
    elif measurement == "process_tree_ps_short_lived_no_sample":
        if peak_rss_bytes != 0:
            raise ContractError(f"{context} short-lived RSS evidence must have a zero peak")
    elif measurement == "unavailable":
        if peak_rss_bytes is not None or not memory_limit_exceeded:
            raise ContractError(f"{context} unavailable RSS evidence must fail closed")
        if completed:
            raise ContractError(f"{context} cannot complete without RSS evidence")
    else:  # Closed callers validate the enum before reaching this branch.
        raise ContractError(f"{context} has an unsupported RSS measurement")


def execute_paired_plan(
    *,
    repository_root: Path,
    plan: object,
    plan_ref: ArtifactRef,
    state_path: str,
    manifest_path: str,
    runner: CommandRunner,
    retry_quarantined: bool = False,
    now: Callable[[], str] | None = None,
) -> dict[str, object]:
    """Serialize one durable state authority so job claims cannot be duplicated."""

    state_file = contained_path(repository_root, validate_relative_path(state_path))
    try:
        with stable_directory_lock(
            state_file.parent,
            create=True,
            exclusive=True,
            nonblocking=True,
        ):
            return _execute_paired_plan_locked(
                repository_root=repository_root,
                plan=plan,
                plan_ref=plan_ref,
                state_path=state_path,
                manifest_path=manifest_path,
                runner=runner,
                retry_quarantined=retry_quarantined,
                now=now,
            )
    except BlockingIOError as error:
        raise ContractError("another process owns this paired-plan state directory") from error


def _execute_paired_plan_locked(
    *,
    repository_root: Path,
    plan: object,
    plan_ref: ArtifactRef,
    state_path: str,
    manifest_path: str,
    runner: CommandRunner,
    retry_quarantined: bool = False,
    now: Callable[[], str] | None = None,
) -> dict[str, object]:
    """Run independent pairs in bounded batches and persist every completed batch."""

    root = validate_paired_plan(plan)
    if isinstance(runner, CommandRunner) and runner.require_clean_repository:
        require_clean_head(repository_root, str(root["gitCommit"]))
    if (
        isinstance(runner, CommandRunner)
        and runner.require_engine_build_receipt
        and root.get("engineBuildReceipt") is None
    ):
        raise ContractError("production paired execution requires an engine build receipt")
    if canonical_sha256(
        {key: value for key, value in root.items() if key != "planSha256"}
    ) != root.get("planSha256"):
        raise ContractError("paired plan self-hash mismatch")
    # The embedded hash covers canonical plan content without itself; the file hash covers the
    # complete serialized document. Both identities are independently verified.
    verify_artifact_ref(repository_root, plan_ref)
    _validate_paired_plan_artifacts(
        root,
        repository_root=repository_root,
        runtime_authorization=(
            isinstance(runner, CommandRunner) and runner.require_engine_build_receipt
        ),
    )
    for key in (
        "engine",
        "modelRegistry",
        "config",
        "startPositions",
        "startPositionValidation",
        "datasetManifest",
    ):
        verify_artifact_ref(
            repository_root, ArtifactRef.from_dict(root.get(key), f"paired plan.{key}")
        )
    receipt_raw = root.get("engineBuildReceipt")
    if receipt_raw is not None:
        verify_artifact_ref(
            repository_root,
            ArtifactRef.from_dict(receipt_raw, "paired plan.engineBuildReceipt"),
        )
    for key in ("champion", "challenger"):
        model_raw = root.get(key)
        if model_raw is None:
            continue
        model = require_mapping(model_raw, f"paired plan.{key}")
        verify_artifact_ref(
            repository_root,
            ArtifactRef.from_dict(model.get("artifact"), f"paired plan.{key}.artifact"),
        )
    state_file = contained_path(repository_root, state_path)
    manifest_file = contained_path(repository_root, manifest_path)
    timestamp = now or _utc_now
    jobs = require_list(root, "jobs", "paired plan", maximum_items=5_000)
    state = _load_or_initialize_state(
        state_file,
        str(root["planSha256"]),
        repository_root=repository_root,
        plan=root,
    )
    initial_state_status = str(state["status"])
    state_revision = int(state["revision"])
    attempts = list(state["attempts"])
    latest = _latest_attempts(attempts)
    pending: list[tuple[Mapping[str, Any], int]] = []
    for job in jobs:
        table = require_mapping(job, "paired plan job")
        latest_attempt = latest.get(str(table["jobId"]))
        if latest_attempt is not None and latest_attempt["status"] == "running":
            pending.append((table, int(latest_attempt["attempt"])))
        elif latest_attempt is None or (
            latest_attempt["status"] == "quarantined" and retry_quarantined
        ):
            pending.append((table, _next_attempt_number(attempts, str(table["jobId"]))))
    max_workers = require_int(root, "maxWorkers", "paired plan", minimum=1, maximum=4)
    state_is_terminal = initial_state_status in {"completed", "completed_with_quarantine"}
    if pending or not state_is_terminal:
        state_revision = _replace_job_state_cas(
            state_file,
            expected_revision=state_revision,
            plan_sha256=str(root["planSha256"]),
            status="running",
            attempts=attempts,
        )
    try:
        for offset in range(0, len(pending), max_workers):
            batch = pending[offset : offset + max_workers]
            for job, attempt_number in batch:
                if not any(
                    row["jobId"] == job["jobId"]
                    and row["attempt"] == attempt_number
                    and row["status"] == "running"
                    for row in attempts
                ):
                    attempts.append(_running_attempt(job, attempt_number, timestamp))
            attempts.sort(key=lambda row: (str(row["jobId"]), int(row["attempt"])))
            state_revision = _replace_job_state_cas(
                state_file,
                expected_revision=state_revision,
                plan_sha256=str(root["planSha256"]),
                status="running",
                attempts=attempts,
            )
            with ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix="phase6-pair"
            ) as pool:
                futures = {
                    pool.submit(
                        _execute_job,
                        repository_root,
                        job,
                        runner,
                        attempt_number,
                        timestamp,
                        root,
                    ): (str(job["jobId"]), attempt_number)
                    for job, attempt_number in batch
                }
                errors: list[BaseException] = []
                for future in as_completed(futures):
                    try:
                        completed_attempt = future.result()
                        job_id, attempt_number = futures[future]
                        matching = [
                            index
                            for index, row in enumerate(attempts)
                            if row["jobId"] == job_id and row["attempt"] == attempt_number
                        ]
                        if len(matching) != 1 or attempts[matching[0]]["status"] != "running":
                            raise ContractError("paired attempt journal entry disappeared")
                        attempts[matching[0]] = completed_attempt
                    except BaseException as error:
                        errors.append(error)
                    else:
                        attempts.sort(key=lambda row: (str(row["jobId"]), int(row["attempt"])))
                        state_revision = _replace_job_state_cas(
                            state_file,
                            expected_revision=state_revision,
                            plan_sha256=str(root["planSha256"]),
                            status="running",
                            attempts=attempts,
                        )
                if errors:
                    raise errors[0]
    except BaseException:
        _replace_job_state_cas(
            state_file,
            expected_revision=state_revision,
            plan_sha256=str(root["planSha256"]),
            status="interrupted",
            attempts=attempts,
        )
        raise

    latest = _latest_attempts(attempts)
    completed_jobs = sum(row["status"] == "completed" for row in latest.values())
    quarantined_jobs = sum(row["status"] == "quarantined" for row in latest.values())
    final_status = "completed" if completed_jobs == len(jobs) else "completed_with_quarantine"
    if pending or initial_state_status != final_status:
        _replace_job_state_cas(
            state_file,
            expected_revision=state_revision,
            plan_sha256=str(root["planSha256"]),
            status=final_status,
            attempts=attempts,
        )
    manifest_schema = (
        SELFPLAY_MANIFEST_SCHEMA
        if root["schema"] == SELFPLAY_PLAN_SCHEMA
        else ARENA_EXECUTION_MANIFEST_SCHEMA
    )
    manifest: dict[str, object] = {
        "schema": manifest_schema,
        "generationId": root["generationId"],
        "plan": plan_ref.as_dict(),
        "planSha256": root["planSha256"],
        "status": final_status,
        "gameCountPlanned": root["gameCount"],
        "jobsPlanned": len(jobs),
        "jobsCompleted": completed_jobs,
        "jobsQuarantined": quarantined_jobs,
        "gamesCompleted": completed_jobs * 2,
        "gamesQuarantined": quarantined_jobs * 2,
        "quarantinedAttempts": sum(row["status"] == "quarantined" for row in attempts),
        "attempts": attempts,
    }
    manifest["manifestSha256"] = canonical_sha256(manifest)
    if final_status == "completed":
        if manifest_file.exists():
            existing = load_json(manifest_file)
            if existing != manifest:
                raise ContractError("existing execution manifest differs from resumed result")
        else:
            write_json_new(manifest_file, manifest)
    return manifest


def validate_paired_plan(raw: object) -> Mapping[str, Any]:
    root = require_mapping(raw, "paired plan")
    schema = root.get("schema")
    if not isinstance(schema, str) or schema not in _PLAN_KEYS:
        raise ContractError("unsupported paired plan schema")
    require_exact_keys(root, _PLAN_KEYS[schema], "paired plan")
    generation_id = require_identifier(root, "generationId", "paired plan")
    require_sha256(root, "configSha256", "paired plan")
    plan_sha256 = require_sha256(root, "planSha256", "paired plan")
    without_hash = dict(root)
    without_hash.pop("planSha256")
    if canonical_sha256(without_hash) != plan_sha256:
        raise ContractError("paired plan self-hash mismatch")
    git_commit = require_string(root, "gitCommit", "paired plan", maximum_length=64)
    if not 7 <= len(git_commit) <= 64 or any(
        character not in "0123456789abcdef" for character in git_commit
    ):
        raise ContractError("paired plan git commit must be a lowercase hexadecimal object ID")
    games = require_int(root, "gameCount", "paired plan", minimum=40, maximum=100)
    pairs = require_int(root, "pairCount", "paired plan", minimum=20, maximum=50)
    if games != pairs * 2:
        raise ContractError("paired plan game and pair counts disagree")
    if games != PHASE6_GAMES or pairs != PHASE6_GAMES // 2:
        raise ContractError("Phase 6 paired plans must contain exactly 40 games")
    normal_pairs = require_int(root, "normalStartPairs", "paired plan", minimum=1, maximum=49)
    start_set_pairs = require_int(root, "startSetPairs", "paired plan", minimum=1, maximum=49)
    if (normal_pairs, start_set_pairs) != (
        PHASE6_NORMAL_START_PAIRS,
        PHASE6_START_SET_PAIRS,
    ):
        raise ContractError("Phase 6 requires ten initial and ten start-set pairs")
    seed = require_int(
        root,
        "seed",
        "paired plan",
        minimum=0,
        maximum=9_007_199_254_740_991,
    )
    if seed != PHASE6_SEED:
        raise ContractError(f"paired plan seed must be {PHASE6_SEED}")
    nodes_per_move = require_int(
        root, "nodesPerMove", "paired plan", minimum=1, maximum=1_000_000_000
    )
    if nodes_per_move != PHASE6_NODES_PER_MOVE:
        raise ContractError(f"paired plan nodesPerMove must be {PHASE6_NODES_PER_MOVE}")
    workers = require_int(root, "maxWorkers", "paired plan", minimum=1, maximum=4)
    if workers != PHASE6_WORKERS:
        raise ContractError(f"paired plan maxWorkers must be {PHASE6_WORKERS}")
    memory = require_int(root, "memoryLimitMiB", "paired plan", minimum=256, maximum=8 * 1024)
    if memory != MAX_WORKING_MEMORY_MIB:
        raise ContractError(f"paired plan memoryLimitMiB must be {MAX_WORKING_MEMORY_MIB}")
    per_worker = require_int(
        root, "memoryPerWorkerMiB", "paired plan", minimum=128, maximum=20 * 1024
    )
    if workers * per_worker > memory:
        raise ContractError("paired plan worker memory exceeds its limit")
    champion = _validate_model_spec(root.get("champion"), "paired plan.champion")
    challenger = (
        _validate_model_spec(root.get("challenger"), "paired plan.challenger")
        if schema == ARENA_PLAN_SCHEMA
        else champion
    )
    if champion["evaluatorKind"] != "neural" or challenger["evaluatorKind"] != "neural":
        raise ContractError("Phase 6 paired plans require neural evaluator models")
    if schema == ARENA_PLAN_SCHEMA and challenger["modelId"] == champion["modelId"]:
        raise ContractError("paired arena champion and challenger must differ")
    for key in (
        "engine",
        "modelRegistry",
        "config",
        "startPositions",
        "startPositionValidation",
        "datasetManifest",
    ):
        ArtifactRef.from_dict(root.get(key), f"paired plan.{key}")
    receipt_raw = root.get("engineBuildReceipt")
    if receipt_raw is not None:
        ArtifactRef.from_dict(receipt_raw, "paired plan.engineBuildReceipt")
    jobs = require_list(root, "jobs", "paired plan", minimum_items=pairs, maximum_items=pairs)
    engine = ArtifactRef.from_dict(root.get("engine"), "paired plan.engine")
    identifiers: set[str] = set()
    game_ids: set[str] = set()
    start_set_ids: set[str] = set()
    start_set_sfens: set[str] = set()
    start_group_counts = {"initial": 0, "start_set": 0}
    run_root: str | None = None
    kind = "selfplay" if schema == SELFPLAY_PLAN_SCHEMA else "arena"
    for index, raw_job in enumerate(jobs):
        context = f"paired plan.jobs[{index}]"
        job = require_mapping(raw_job, context)
        require_exact_keys(job, _JOB_KEYS, context)
        job_id = require_identifier(job, "jobId", context)
        if job_id in identifiers:
            raise ContractError(f"duplicate paired-plan job ID: {job_id}")
        identifiers.add(job_id)
        if job_id != f"pair-{index:04d}":
            raise ContractError("paired-plan job IDs must match their contiguous pair index")
        if require_int(job, "pairIndex", context, minimum=0, maximum=49) != index:
            raise ContractError("paired-plan pair indices must be contiguous")
        start_group = require_enum(job, "startGroup", context, {"initial", "start_set"})
        expected_group = "initial" if index < normal_pairs else "start_set"
        if start_group != expected_group:
            raise ContractError(
                "paired-plan start groups must use initial pairs before start-set pairs"
            )
        start_group_counts[start_group] += 1
        start_position_id = require_identifier(job, "startPositionId", context)
        sfen = require_string(job, "sfen", context, maximum_length=512)
        _validate_plan_sfen(sfen, context)
        if start_group == "initial":
            if start_position_id != "standard-initial" or sfen != INITIAL_SFEN:
                raise ContractError("initial paired jobs must use the standard initial position")
        else:
            if start_position_id == "standard-initial":
                raise ContractError(
                    "start-set jobs must reference an approved non-initial position"
                )
            if start_position_id in start_set_ids or sfen in start_set_sfens:
                raise ContractError("start-set paired jobs must use unique approved positions")
            start_set_ids.add(start_position_id)
            start_set_sfens.add(sfen)
        job_seed = require_int(job, "seed", context, minimum=0, maximum=9_007_199_254_740_991)
        expected_seed = _derive_job_seed(seed, kind, index, start_position_id)
        if job_seed != expected_seed:
            raise ContractError(f"{context}.seed disagrees with deterministic derivation")
        job_game_ids = _validate_string_array(job, "gameIds", context, expected=2, identifiers=True)
        expected_game_ids = (f"game-{index * 2:06d}", f"game-{index * 2 + 1:06d}")
        if job_game_ids != expected_game_ids or any(value in game_ids for value in job_game_ids):
            raise ContractError("paired-plan game IDs must be unique and contiguous")
        game_ids.update(job_game_ids)
        colors = _validate_string_array(job, "modelAColorOrder", context, expected=2)
        if colors != ("black", "white"):
            raise ContractError("paired plan must swap model A from black to white")
        output_dir = require_relative_path(job, "outputDir", context)
        output_path = PurePosixPath(output_dir)
        if output_path.name != job_id or output_path.parent.name != "jobs":
            raise ContractError(f"{context}.outputDir must use the generation jobs layout")
        candidate_run_root = output_path.parent.parent.as_posix()
        if run_root is None:
            run_root = candidate_run_root
            expected_suffix = f"/{generation_id}/{kind}"
            if not run_root.endswith(expected_suffix):
                raise ContractError("paired-plan output root does not bind generation and kind")
        elif candidate_run_root != run_root:
            raise ContractError("paired-plan jobs do not share one run root")
        report_path = require_relative_path(job, "reportPath", context)
        quarantine_path = require_relative_path(job, "quarantinePath", context)
        csa_paths = _validate_string_array(job, "csaPaths", context, expected=2, paths=True)
        if report_path != f"{output_dir}/arena-report.json":
            raise ContractError(f"{context}.reportPath disagrees with its output directory")
        if csa_paths != (
            f"{output_dir}/games/game-000001.csa",
            f"{output_dir}/games/game-000002.csa",
        ):
            raise ContractError(f"{context}.csaPaths disagree with its output directory")
        if quarantine_path != f"{run_root}/quarantine/{job_id}.json":
            raise ContractError(f"{context}.quarantinePath disagrees with its run root")
        command = require_mapping(job.get("command"), f"{context}.command")
        require_exact_keys(command, _COMMAND_KEYS, f"{context}.command")
        require_enum(command, "kind", f"{context}.command", {"engine_arena"})
        argv = _command_argv(command, "engine_arena")
        if argv[0] != engine.path:
            raise ContractError("paired job command does not use the hashed plan engine")
        _validate_arena_argv(
            argv,
            job=job,
            model_a=challenger,
            model_b=champion,
            git_commit=git_commit,
            nodes_per_move=nodes_per_move,
            context=context,
        )
        require_int(
            command,
            "timeoutSeconds",
            f"{context}.command",
            minimum=1,
            maximum=172_800,
        )
    if start_group_counts != {"initial": normal_pairs, "start_set": start_set_pairs}:
        raise ContractError("paired-plan start-group counts disagree with its partition")
    return root


def validate_execution_manifest(
    raw: object,
    *,
    repository_root: Path,
    manifest_ref: ArtifactRef,
    plan: object,
    plan_ref: ArtifactRef,
) -> Mapping[str, Any]:
    """Validate one completed immutable paired execution and every artifact it names."""

    plan_root = validate_paired_plan(plan)
    verify_artifact_ref(repository_root, plan_ref)
    verify_artifact_ref(repository_root, manifest_ref)
    _validate_paired_plan_artifacts(plan_root, repository_root=repository_root)
    root = require_mapping(raw, "paired execution manifest")
    require_exact_keys(
        root,
        {
            "schema",
            "generationId",
            "plan",
            "planSha256",
            "status",
            "gameCountPlanned",
            "jobsPlanned",
            "jobsCompleted",
            "jobsQuarantined",
            "gamesCompleted",
            "gamesQuarantined",
            "quarantinedAttempts",
            "attempts",
            "manifestSha256",
        },
        "paired execution manifest",
    )
    expected_schema = (
        SELFPLAY_MANIFEST_SCHEMA
        if plan_root.get("schema") == SELFPLAY_PLAN_SCHEMA
        else ARENA_EXECUTION_MANIFEST_SCHEMA
    )
    if root.get("schema") != expected_schema:
        raise ContractError("execution manifest schema does not match its plan")
    if root.get("generationId") != plan_root.get("generationId"):
        raise ContractError("execution manifest generation does not match its plan")
    if ArtifactRef.from_dict(root.get("plan"), "paired execution manifest.plan") != plan_ref:
        raise ContractError("execution manifest references a different plan artifact")
    if root.get("planSha256") != plan_root.get("planSha256"):
        raise ContractError("execution manifest plan hash does not match its plan")
    if root.get("status") != "completed":
        raise ContractError("downstream processing requires a completed paired execution")
    jobs = require_list(plan_root, "jobs", "paired plan", maximum_items=5_000)
    job_ids = {str(job["jobId"]) for job in jobs}
    attempts_raw = require_list(
        root,
        "attempts",
        "paired execution manifest",
        minimum_items=len(jobs),
        maximum_items=20_000,
    )
    attempts = _validate_attempt_records(
        attempts_raw,
        repository_root=repository_root,
        plan=plan_root,
        context="paired execution manifest",
    )
    latest = _latest_attempts(attempts)
    if set(latest) != job_ids or any(row.get("status") != "completed" for row in latest.values()):
        raise ContractError("execution manifest lacks a final completed attempt for every job")
    for raw_job in jobs:
        job = require_mapping(raw_job, "paired plan job")
        job_id = str(job["jobId"])
        attempt = latest[job_id]
        report = ArtifactRef.from_dict(attempt.get("report"), f"attempt {job_id}.report")
        if report.path != job.get("reportPath"):
            raise ContractError(f"execution report path differs from plan job {job_id}")
        csa_values = require_list(
            attempt, "csa", f"attempt {job_id}", minimum_items=2, maximum_items=2
        )
        csa_refs = [
            ArtifactRef.from_dict(value, f"attempt {job_id}.csa[{index}]")
            for index, value in enumerate(csa_values)
        ]
        if [reference.path for reference in csa_refs] != job.get("csaPaths"):
            raise ContractError(f"execution CSA paths differ from plan job {job_id}")
        from .arena import validate_phase6_pair_report_binding

        champion = require_mapping(plan_root.get("champion"), "paired plan.champion")
        model_a = (
            require_mapping(plan_root.get("challenger"), "paired plan.challenger")
            if plan_root.get("schema") == ARENA_PLAN_SCHEMA
            else champion
        )
        validate_phase6_pair_report_binding(
            load_json_artifact(repository_root, report),
            repository_root=repository_root,
            job=job,
            git_commit=str(plan_root["gitCommit"]),
            nodes_per_move=int(plan_root["nodesPerMove"]),
            model_a=model_a,
            model_b=champion,
            csa_refs=(csa_refs[0], csa_refs[1]),
            context=f"paired execution report {job_id}",
        )
    planned_games = require_int(
        root, "gameCountPlanned", "paired execution manifest", minimum=40, maximum=100
    )
    if planned_games != plan_root.get("gameCount"):
        raise ContractError("execution manifest planned game count disagrees with its plan")
    expected_counts = {
        "jobsPlanned": len(jobs),
        "jobsCompleted": len(jobs),
        "jobsQuarantined": 0,
        "gamesCompleted": planned_games,
        "gamesQuarantined": 0,
        "quarantinedAttempts": sum(row.get("status") == "quarantined" for row in attempts),
    }
    for key, expected in expected_counts.items():
        if (
            require_int(
                root,
                key,
                "paired execution manifest",
                minimum=0,
                maximum=20_000,
            )
            != expected
        ):
            raise ContractError(f"execution manifest {key} is inconsistent")
    expected_hash = require_sha256(root, "manifestSha256", "paired execution manifest")
    without_hash = dict(root)
    without_hash.pop("manifestSha256")
    if canonical_sha256(without_hash) != expected_hash:
        raise ContractError("execution manifest self-hash mismatch")
    return root


def _validate_paired_plan_artifacts(
    plan: Mapping[str, Any],
    *,
    repository_root: Path,
    runtime_authorization: bool = False,
) -> None:
    """Rebuild approved inputs; optionally prove the current executable authority."""

    from .config import parse_selfplay_config_bytes
    from .planning import (
        _select_start_positions,
        validate_start_position_validation,
    )
    from .registry import validate_model_registry
    from .starts import validate_start_position_source_artifact

    starts_ref = ArtifactRef.from_dict(plan.get("startPositions"), "paired plan.startPositions")
    validation_ref = ArtifactRef.from_dict(
        plan.get("startPositionValidation"), "paired plan.startPositionValidation"
    )
    engine_ref = ArtifactRef.from_dict(plan.get("engine"), "paired plan.engine")
    receipt_raw = plan.get("engineBuildReceipt")
    engine_receipt = (
        ArtifactRef.from_dict(receipt_raw, "paired plan.engineBuildReceipt")
        if receipt_raw is not None
        else None
    )
    dataset_ref = ArtifactRef.from_dict(plan.get("datasetManifest"), "paired plan.datasetManifest")
    config_ref = ArtifactRef.from_dict(plan.get("config"), "paired plan.config")
    config = parse_selfplay_config_bytes(
        load_bytes_artifact(repository_root, config_ref, maximum_bytes=64 * 1024),
        config_ref.path,
    )
    if plan.get("configSha256") != config.sha256:
        raise ContractError("paired plan semantic config hash differs from its config artifact")
    expected_config_values = {
        "gameCount": config.run.games,
        "pairCount": config.run.games // 2,
        "normalStartPairs": config.run.normal_start_pairs,
        "startSetPairs": config.run.start_set_pairs,
        "seed": config.run.seed,
        "nodesPerMove": config.run.nodes_per_move,
        "maxWorkers": config.resources.workers,
        "memoryLimitMiB": config.resources.memory_limit_mib,
        "memoryPerWorkerMiB": config.resources.memory_per_worker_mib,
    }
    for key, expected in expected_config_values.items():
        if plan.get(key) != expected:
            raise ContractError(f"paired plan {key} differs from its config artifact")
    from .engine_receipt import DEFAULT_ENGINE_PATH

    if config.paths.engine_cli != DEFAULT_ENGINE_PATH:
        raise ContractError("paired config engineCli must remain the operator alias")
    if engine_receipt is not None:
        from .engine_receipt import (
            validate_engine_build_receipt,
            validate_engine_build_receipt_document,
        )

        receipt = load_json_artifact(repository_root, engine_receipt)
        validate_engine_build_receipt_document(
            receipt,
            expected_engine=engine_ref,
            expected_git_commit=str(plan["gitCommit"]),
        )
        if runtime_authorization:
            validate_engine_build_receipt(
                repository_root,
                engine_receipt,
                expected_engine=engine_ref,
                expected_git_commit=str(plan["gitCommit"]),
            )
    elif runtime_authorization:
        raise ContractError("paired runtime authorization requires an immutable engine receipt")
    if starts_ref.path != config.paths.start_positions_manifest:
        raise ContractError("paired plan start-position path differs from its config artifact")
    registry_ref = ArtifactRef.from_dict(plan.get("modelRegistry"), "paired plan.modelRegistry")
    registry = validate_model_registry(
        load_json_artifact(repository_root, registry_ref),
        repository_root=repository_root,
    )
    champion = require_mapping(plan.get("champion"), "paired plan.champion")
    _validate_plan_registry_model(
        champion,
        registry,
        expected_id=registry.get("championModelId"),
        context="paired plan.champion",
    )
    if plan.get("schema") == SELFPLAY_PLAN_SCHEMA:
        if registry.get("challengerModelId") is not None:
            raise ContractError("self-play plan registry already has an active challenger")
    else:
        challenger = require_mapping(plan.get("challenger"), "paired plan.challenger")
        _validate_plan_registry_model(
            challenger,
            registry,
            expected_id=registry.get("challengerModelId"),
            context="paired plan.challenger",
        )
        generations = require_list(
            registry, "generations", "paired plan.modelRegistry", maximum_items=10_000
        )
        active = [
            generation
            for generation in generations
            if isinstance(generation, dict) and generation.get("status") == "arena"
        ]
        if len(active) != 1 or active[0].get("generationId") != plan.get("generationId"):
            raise ContractError("arena plan generation is not the active registry generation")
    starts_raw = load_json_artifact(repository_root, starts_ref)
    starts = validate_start_position_source_artifact(
        repository_root=repository_root,
        start_positions=starts_raw,
        start_positions_ref=starts_ref,
    )
    if starts.dataset_manifest != dataset_ref:
        raise ContractError("paired plan dataset differs from its approved Phase 3 starts")
    validation_raw = load_json_artifact(repository_root, validation_ref)
    validate_start_position_validation(
        validation_raw,
        start_positions_ref=starts_ref,
        engine_ref=engine_ref,
        positions=starts,
        repository_root=repository_root,
        engine_build_receipt=engine_receipt,
        runtime_authorization=runtime_authorization,
    )
    validation = require_mapping(validation_raw, "paired plan start-position validation")
    if validation.get("gitCommit") != plan.get("gitCommit"):
        raise ContractError("start-position validation git commit differs from its paired plan")

    schema = str(plan["schema"])
    split = "train" if schema == SELFPLAY_PLAN_SCHEMA else "validation"
    selection_seed = int(plan["seed"])
    if schema == ARENA_PLAN_SCHEMA:
        selection_seed ^= 0x4152_454E_41
    selected = _select_start_positions(
        tuple(position for position in starts.positions if position.split == split),
        count=int(plan["startSetPairs"]),
        seed=selection_seed,
    )
    jobs = require_list(plan, "jobs", "paired plan", maximum_items=5_000)
    kind = "selfplay" if schema == SELFPLAY_PLAN_SCHEMA else "arena"
    expected_run_root = f"{config.paths.output_root}/{plan['generationId']}/{kind}"
    for index, raw_job in enumerate(jobs):
        job = require_mapping(raw_job, f"paired plan.jobs[{index}]")
        if job.get("outputDir") != f"{expected_run_root}/jobs/pair-{index:04d}":
            raise ContractError("paired plan output directories differ from its config artifact")
        command = require_mapping(job.get("command"), f"paired plan.jobs[{index}].command")
        if command.get("timeoutSeconds") != config.resources.game_timeout_seconds:
            raise ContractError("paired plan timeout differs from its config artifact")
        argv = _command_argv(command, "engine_arena")
        values, _ = _parse_closed_options(
            argv,
            prefix=(argv[0], "arena"),
            required={
                "--games",
                "--player-a",
                "--player-b",
                "--a-depth",
                "--b-depth",
                "--a-hash-mb",
                "--b-hash-mb",
                "--nodes",
                "--max-plies",
                "--seed",
                "--sfen",
                "--git-commit",
                "--output-dir",
            },
            optional={"--a-model", "--b-model"},
            context=f"paired plan.jobs[{index}].command",
        )
        expected_options = {
            "--a-depth": str(config.run.search_depth),
            "--b-depth": str(config.run.search_depth),
            "--a-hash-mb": str(config.run.hash_mib),
            "--b-hash-mb": str(config.run.hash_mib),
            "--max-plies": str(config.run.max_plies),
        }
        if any(values.get(option) != expected for option, expected in expected_options.items()):
            raise ContractError("paired plan search options differ from its config artifact")
    start_jobs = jobs[int(plan["normalStartPairs"]) :]
    observed = [
        (str(job["startPositionId"]), str(job["sfen"]))
        for job in start_jobs
        if isinstance(job, dict)
    ]
    expected = [(position.position_id, position.sfen) for position in selected]
    if observed != expected:
        raise ContractError(
            f"paired plan start-set jobs are not the deterministic {split} selection"
        )


def _validate_plan_registry_model(
    planned: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    expected_id: object,
    context: str,
) -> None:
    model_id = require_identifier(planned, "modelId", context)
    if model_id != expected_id:
        raise ContractError(f"{context} is not the registry lifecycle model")
    models = require_list(registry, "models", "model registry", maximum_items=10_000)
    registered = next(
        (
            require_mapping(model, "registered model")
            for model in models
            if isinstance(model, dict) and model.get("modelId") == model_id
        ),
        None,
    )
    if registered is None:
        raise ContractError(f"{context} is absent from the model registry")
    planned_artifact = ArtifactRef.from_dict(planned.get("artifact"), f"{context}.artifact")
    registered_artifact = ArtifactRef.from_dict(
        registered.get("artifact"), f"{context}.registeredArtifact"
    )
    if planned_artifact != registered_artifact or planned.get("evaluatorKind") != registered.get(
        "evaluatorKind"
    ):
        raise ContractError(f"{context} identity differs from the model registry")


def _execute_job(
    repository_root: Path,
    job: Mapping[str, Any],
    runner: CommandRunner,
    attempt: int,
    now: Callable[[], str],
    plan: Mapping[str, Any],
) -> dict[str, object]:
    job_id = str(job["jobId"])
    output_dir = str(job["outputDir"])
    if "/jobs/" not in output_dir:
        raise ContractError("paired job output directory lacks the required jobs component")
    run_root = output_dir.rsplit("/jobs/", 1)[0]
    log_root = f"{run_root}/logs/{job_id}"
    evidence_path = f"{log_root}/attempt-{attempt:03d}.evidence.json"
    evidence_file = contained_path(repository_root, evidence_path)
    if evidence_file.exists():
        evidence_ref = artifact_ref(repository_root, evidence_path)
        evidence = _validate_attempt_command_receipt(
            load_json_artifact(repository_root, evidence_ref),
            repository_root=repository_root,
            plan=plan,
            job=job,
            attempt_number=attempt,
            context=f"paired attempt evidence {job_id}/{attempt}",
        )
        return _attempt_record_from_evidence(evidence, evidence_ref)
    receipt_path = f"{log_root}/attempt-{attempt:03d}.receipt.json"
    if contained_path(repository_root, receipt_path).exists():
        receipt_ref = artifact_ref(repository_root, receipt_path)
        receipt = require_mapping(
            load_json_artifact(repository_root, receipt_ref), "command receipt"
        )
        stdout_path = ArtifactRef.from_dict(receipt.get("stdout"), "command receipt.stdout").path
        stderr_path = ArtifactRef.from_dict(receipt.get("stderr"), "command receipt.stderr").path
    else:
        stdout_path, stderr_path = _available_attempt_logs(repository_root, log_root, attempt)
    resume = contained_path(repository_root, f"{output_dir}/arena.state").is_file()
    outcome = runner.run(
        job["command"],
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        resume=resume,
        expected_executable=ArtifactRef.from_dict(plan["engine"], "paired plan.engine"),
        engine_build_receipt=(
            ArtifactRef.from_dict(plan["engineBuildReceipt"], "paired plan.engineBuildReceipt")
            if plan.get("engineBuildReceipt") is not None
            else None
        ),
        memory_limit_mib=int(plan["memoryPerWorkerMiB"]),
        receipt_path=receipt_path,
    )
    process_receipt = outcome.command_receipt
    if process_receipt is None:
        expected_engine = ArtifactRef.from_dict(plan["engine"], "paired plan.engine")
        planned_engine_receipt = _optional_artifact_ref(
            plan.get("engineBuildReceipt"), "paired plan.engineBuildReceipt"
        )
        invocation = {
            "command": dict(require_mapping(job["command"], "paired job.command")),
            "resume": resume,
            "memoryLimitMiB": int(plan["memoryPerWorkerMiB"]),
            "expectedExecutable": expected_engine.as_dict(),
            "engineBuildReceipt": (
                None if planned_engine_receipt is None else planned_engine_receipt.as_dict()
            ),
            "runtimeReceipt": None,
        }
        synthetic_receipt = {
            "schema": _PROCESS_RECEIPT_SCHEMA,
            **invocation,
            "commandSha256": canonical_sha256(invocation),
            "returnCode": outcome.return_code,
            "timedOut": outcome.timed_out,
            "outputLimitExceeded": outcome.output_limit_exceeded,
            "memoryLimitExceeded": outcome.memory_limit_exceeded,
            "peakRssBytes": outcome.peak_rss_bytes,
            "rssMeasurement": outcome.rss_measurement,
            "stdout": outcome.stdout.as_dict(),
            "stderr": outcome.stderr.as_dict(),
        }
        write_json_new(contained_path(repository_root, receipt_path), synthetic_receipt)
        process_receipt = artifact_ref(repository_root, receipt_path)
    failure_category: str | None = None
    failure_detail: str | None = None
    report: ArtifactRef | None = None
    csa: list[ArtifactRef] = []
    observed_report: ArtifactRef | None = None
    observed_csa: list[ArtifactRef] = []
    if outcome.timed_out:
        failure_category = "timeout"
    elif outcome.memory_limit_exceeded:
        failure_category = "memory_limit"
    elif outcome.output_limit_exceeded:
        failure_category = "output_limit"
    elif outcome.return_code != 0:
        failure_category = "nonzero_exit"
    else:
        try:
            report = artifact_ref(repository_root, str(job["reportPath"]))
            csa = [artifact_ref(repository_root, str(path)) for path in job["csaPaths"]]
            observed_report = report
            observed_csa = list(csa)
            from .arena import validate_phase6_pair_report_binding

            champion = require_mapping(plan.get("champion"), "paired plan.champion")
            model_a = (
                require_mapping(plan.get("challenger"), "paired plan.challenger")
                if plan.get("schema") == ARENA_PLAN_SCHEMA
                else champion
            )
            validate_phase6_pair_report_binding(
                load_json_artifact(repository_root, report),
                repository_root=repository_root,
                job=job,
                git_commit=str(plan["gitCommit"]),
                nodes_per_move=int(plan["nodesPerMove"]),
                model_a=model_a,
                model_b=champion,
                csa_refs=(csa[0], csa[1]),
                context=f"paired execution report {job_id}",
            )
        except ContractError as error:
            if observed_report is None or len(observed_csa) != 2:
                observed_report = None
                observed_csa = []
            report = None
            csa = []
            failure_category = "missing_or_invalid_artifact"
            failure_detail = str(error)[:1_024]
    status = "completed" if failure_category is None else "quarantined"
    completed_at = now()
    record: dict[str, object] = {
        "jobId": job_id,
        "attempt": attempt,
        "status": status,
        "returnCode": outcome.return_code,
        "timedOut": outcome.timed_out,
        "outputLimitExceeded": outcome.output_limit_exceeded,
        "memoryLimitExceeded": outcome.memory_limit_exceeded,
        "peakRssBytes": outcome.peak_rss_bytes,
        "rssMeasurement": outcome.rss_measurement,
        "stdout": outcome.stdout.as_dict(),
        "stderr": outcome.stderr.as_dict(),
        "report": report.as_dict() if report is not None else None,
        "csa": [item.as_dict() for item in csa],
        "quarantine": None,
        "failureCategory": failure_category,
        "completedAt": completed_at,
        "commandReceipt": None,
    }
    if failure_category is not None:
        base = str(job["quarantinePath"])
        quarantine = (
            base if attempt == 1 else base.removesuffix(".json") + f".attempt-{attempt}.json"
        )
        quarantine = _available_quarantine_path(repository_root, quarantine)
        quarantine_file = contained_path(repository_root, quarantine)
        quarantine_payload = {
            "schema": "phase6_quarantined_job/v1",
            "jobId": job_id,
            "attempt": attempt,
            "failureCategory": failure_category,
            "returnCode": outcome.return_code,
            "timedOut": outcome.timed_out,
            "outputLimitExceeded": outcome.output_limit_exceeded,
            "memoryLimitExceeded": outcome.memory_limit_exceeded,
            "peakRssBytes": outcome.peak_rss_bytes,
            "rssMeasurement": outcome.rss_measurement,
            "stdout": outcome.stdout.as_dict(),
            "stderr": outcome.stderr.as_dict(),
            "observedReport": (observed_report.as_dict() if observed_report is not None else None),
            "observedCsa": [item.as_dict() for item in observed_csa],
            "failureDetail": failure_detail,
        }
        write_json_new(quarantine_file, quarantine_payload)
        record["quarantine"] = artifact_ref(repository_root, quarantine).as_dict()
    assert process_receipt is not None
    evidence_payload: dict[str, object] = {
        "schema": _ATTEMPT_RECEIPT_SCHEMA,
        "planSha256": plan["planSha256"],
        "jobId": job_id,
        "attempt": attempt,
        "command": job["command"],
        "engine": plan["engine"],
        "engineBuildReceipt": plan.get("engineBuildReceipt"),
        "processReceipt": process_receipt.as_dict(),
        "result": {
            "returnCode": outcome.return_code,
            "timedOut": outcome.timed_out,
            "outputLimitExceeded": outcome.output_limit_exceeded,
            "memoryLimitExceeded": outcome.memory_limit_exceeded,
            "peakRssBytes": outcome.peak_rss_bytes,
            "rssMeasurement": outcome.rss_measurement,
        },
        "stdout": outcome.stdout.as_dict(),
        "stderr": outcome.stderr.as_dict(),
        "report": record["report"],
        "csa": record["csa"],
        "quarantine": record["quarantine"],
        "failureCategory": failure_category,
        "completedAt": completed_at,
    }
    evidence_payload["receiptSha256"] = canonical_sha256(evidence_payload)
    _validate_attempt_command_receipt(
        evidence_payload,
        repository_root=repository_root,
        plan=plan,
        job=job,
        attempt_number=attempt,
        context=f"paired attempt evidence {job_id}/{attempt}",
    )
    write_json_new(evidence_file, evidence_payload)
    evidence_ref = artifact_ref(repository_root, evidence_path)
    record["commandReceipt"] = evidence_ref.as_dict()
    return record


def _validate_attempt_command_receipt(
    raw: object,
    *,
    repository_root: Path,
    plan: Mapping[str, Any],
    job: Mapping[str, Any],
    attempt_number: int,
    context: str,
) -> dict[str, Any]:
    evidence = dict(require_mapping(raw, context))
    require_exact_keys(evidence, _ATTEMPT_RECEIPT_KEYS, context)
    if evidence.get("schema") != _ATTEMPT_RECEIPT_SCHEMA:
        raise ContractError(f"{context} has an unsupported schema")
    if require_sha256(evidence, "planSha256", context) != plan.get("planSha256"):
        raise ContractError(f"{context} belongs to another plan")
    if require_identifier(evidence, "jobId", context) != job.get("jobId"):
        raise ContractError(f"{context} belongs to another job")
    if require_int(evidence, "attempt", context, minimum=1, maximum=100) != attempt_number:
        raise ContractError(f"{context} has a different attempt number")
    command = require_mapping(evidence.get("command"), f"{context}.command")
    require_exact_keys(command, _COMMAND_KEYS, f"{context}.command")
    if command != job.get("command"):
        raise ContractError(f"{context} records a different command")
    engine = ArtifactRef.from_dict(evidence.get("engine"), f"{context}.engine")
    planned_engine = ArtifactRef.from_dict(plan.get("engine"), "paired plan.engine")
    if engine != planned_engine:
        raise ContractError(f"{context} records a different engine")
    engine_receipt = _optional_artifact_ref(
        evidence.get("engineBuildReceipt"), f"{context}.engineBuildReceipt"
    )
    planned_receipt = _optional_artifact_ref(
        plan.get("engineBuildReceipt"), "paired plan.engineBuildReceipt"
    )
    if engine_receipt != planned_receipt:
        raise ContractError(f"{context} records a different engine build receipt")
    process_receipt = _optional_artifact_ref(
        evidence.get("processReceipt"), f"{context}.processReceipt"
    )
    if process_receipt is None:
        raise ContractError(f"{context} lacks its process evidence")

    result = require_mapping(evidence.get("result"), f"{context}.result")
    require_exact_keys(result, _ATTEMPT_RESULT_KEYS, f"{context}.result")
    return_code = require_int(result, "returnCode", f"{context}.result", minimum=-255, maximum=255)
    timed_out = require_bool(result, "timedOut", f"{context}.result")
    output_limit = require_bool(result, "outputLimitExceeded", f"{context}.result")
    memory_limit = require_bool(result, "memoryLimitExceeded", f"{context}.result")
    peak = result.get("peakRssBytes")
    if peak is not None:
        peak = require_int(
            result,
            "peakRssBytes",
            f"{context}.result",
            minimum=0,
            maximum=2**63 - 1,
        )
    measurement = require_enum(
        result,
        "rssMeasurement",
        f"{context}.result",
        {
            "process_tree_ps_rss_sum",
            "process_tree_ps_short_lived_no_sample",
            "unavailable",
        },
    )
    failure = evidence.get("failureCategory")
    if failure is not None and failure not in {
        "timeout",
        "memory_limit",
        "output_limit",
        "nonzero_exit",
        "missing_or_invalid_artifact",
    }:
        raise ContractError(f"{context} has an invalid failure category")
    expected_process_failure = (
        "timeout"
        if timed_out
        else "memory_limit"
        if memory_limit
        else "output_limit"
        if output_limit
        else "nonzero_exit"
        if return_code != 0
        else None
    )
    if expected_process_failure is not None and failure != expected_process_failure:
        raise ContractError(f"{context} failure category violates precedence")
    if expected_process_failure is None and failure not in {
        None,
        "missing_or_invalid_artifact",
    }:
        raise ContractError(f"{context} failure category disagrees with its result")
    completed = failure is None
    _validate_rss_result(
        measurement=measurement,
        peak_rss_bytes=peak,
        memory_limit_bytes=int(plan["memoryPerWorkerMiB"]) * 1024 * 1024,
        memory_limit_exceeded=memory_limit,
        completed=completed,
        context=context,
    )
    stdout = ArtifactRef.from_dict(evidence.get("stdout"), f"{context}.stdout")
    stderr = ArtifactRef.from_dict(evidence.get("stderr"), f"{context}.stderr")
    verify_artifact_ref(repository_root, stdout)
    verify_artifact_ref(repository_root, stderr)
    report = _optional_artifact_ref(evidence.get("report"), f"{context}.report")
    csa_raw = require_list(evidence, "csa", context, maximum_items=2)
    csa = [
        ArtifactRef.from_dict(value, f"{context}.csa[{index}]")
        for index, value in enumerate(csa_raw)
    ]
    for reference in csa:
        verify_artifact_ref(repository_root, reference)
    quarantine = _optional_artifact_ref(evidence.get("quarantine"), f"{context}.quarantine")
    report_completed: str | None = None
    if completed:
        if report is None or len(csa) != 2 or quarantine is not None:
            raise ContractError(f"{context} completed outputs are incomplete")
        if report.path != job.get("reportPath") or [item.path for item in csa] != job.get(
            "csaPaths"
        ):
            raise ContractError(f"{context} outputs differ from its planned paths")
        verify_artifact_ref(repository_root, report)
        report_root = require_mapping(
            load_json_artifact(repository_root, report), f"{context}.report"
        )
        report_run = require_mapping(report_root.get("run"), f"{context}.report.run")
        report_completed = validate_utc_timestamp(
            report_run.get("completedAt"), f"{context}.report.run.completedAt"
        )
    else:
        if report is not None or csa or quarantine is None:
            raise ContractError(f"{context} quarantined outputs are inconsistent")
        verify_artifact_ref(repository_root, quarantine)
    attempt_completed = validate_utc_timestamp(
        evidence.get("completedAt"), f"{context}.completedAt"
    )
    if completed:
        assert report_completed is not None
        if datetime.fromisoformat(
            attempt_completed.removesuffix("Z") + "+00:00"
        ) < datetime.fromisoformat(report_completed.removesuffix("Z") + "+00:00"):
            raise ContractError(f"{context} completed before its arena report")
    expected_hash = require_sha256(evidence, "receiptSha256", context)
    unsigned = dict(evidence)
    del unsigned["receiptSha256"]
    if canonical_sha256(unsigned) != expected_hash:
        raise ContractError(f"{context} self-hash mismatch")

    if process_receipt is not None:
        process_root = require_mapping(
            load_json_artifact(repository_root, process_receipt),
            f"{context}.processReceipt",
        )
        process_resume = require_bool(process_root, "resume", f"{context}.processReceipt")
        memory_mib = int(plan["memoryPerWorkerMiB"])
        invocation = {
            "command": dict(command),
            "resume": process_resume,
            "memoryLimitMiB": memory_mib,
            "expectedExecutable": engine.as_dict(),
            "engineBuildReceipt": (None if engine_receipt is None else engine_receipt.as_dict()),
            "runtimeReceipt": None,
        }
        process_outcome = validate_command_receipt(
            repository_root,
            process_receipt,
            command=dict(command),
            command_identity=canonical_sha256(invocation),
            resume=process_resume,
            memory_limit_mib=memory_mib,
            expected_executable=engine,
            engine_build_receipt=engine_receipt,
            runtime_receipt=None,
            expected_git_commit=None,
            stdout_path=stdout.path,
            stderr_path=stderr.path,
        )
        observed_result = {
            "returnCode": process_outcome.return_code,
            "timedOut": process_outcome.timed_out,
            "outputLimitExceeded": process_outcome.output_limit_exceeded,
            "memoryLimitExceeded": process_outcome.memory_limit_exceeded,
            "peakRssBytes": process_outcome.peak_rss_bytes,
            "rssMeasurement": process_outcome.rss_measurement,
        }
        if (
            observed_result != result
            or process_outcome.stdout != stdout
            or process_outcome.stderr != stderr
        ):
            raise ContractError(f"{context} differs from its process receipt")
    return evidence


def _attempt_record_from_evidence(
    evidence: Mapping[str, Any], evidence_ref: ArtifactRef
) -> dict[str, object]:
    result = require_mapping(evidence.get("result"), "attempt evidence.result")
    failure = evidence.get("failureCategory")
    return {
        "jobId": evidence["jobId"],
        "attempt": evidence["attempt"],
        "status": "completed" if failure is None else "quarantined",
        "returnCode": result["returnCode"],
        "timedOut": result["timedOut"],
        "outputLimitExceeded": result["outputLimitExceeded"],
        "memoryLimitExceeded": result["memoryLimitExceeded"],
        "peakRssBytes": result["peakRssBytes"],
        "rssMeasurement": result["rssMeasurement"],
        "stdout": evidence["stdout"],
        "stderr": evidence["stderr"],
        "report": evidence["report"],
        "csa": evidence["csa"],
        "quarantine": evidence["quarantine"],
        "failureCategory": failure,
        "completedAt": evidence["completedAt"],
        "commandReceipt": evidence_ref.as_dict(),
    }


def _next_attempt_number(attempts: list[Mapping[str, Any]], job_id: str) -> int:
    number = 1 + sum(str(row["jobId"]) == job_id for row in attempts)
    if number > 100:
        raise ContractError(f"paired job exhausted its retry limit: {job_id}")
    return number


def _running_attempt(
    job: Mapping[str, Any], attempt: int, now: Callable[[], str]
) -> dict[str, object]:
    return {
        "jobId": str(job["jobId"]),
        "attempt": attempt,
        "status": "running",
        "returnCode": None,
        "timedOut": None,
        "outputLimitExceeded": None,
        "memoryLimitExceeded": None,
        "peakRssBytes": None,
        "rssMeasurement": None,
        "stdout": None,
        "stderr": None,
        "report": None,
        "csa": [],
        "quarantine": None,
        "failureCategory": None,
        "completedAt": now(),
        "commandReceipt": None,
    }


def _available_attempt_logs(repository_root: Path, log_root: str, attempt: int) -> tuple[str, str]:
    base = f"{log_root}/attempt-{attempt:03d}"
    candidates = [base, *(f"{base}.recovery-{index:03d}" for index in range(1, 101))]
    for candidate in candidates:
        stdout_path = f"{candidate}.stdout.log"
        stderr_path = f"{candidate}.stderr.log"
        stdout = contained_path(repository_root, stdout_path)
        stderr = contained_path(repository_root, stderr_path)
        if (
            not stdout.exists()
            and not stdout.is_symlink()
            and not stderr.exists()
            and not stderr.is_symlink()
        ):
            return stdout_path, stderr_path
    raise ContractError("paired job exhausted its immutable recovery log paths")


def _available_quarantine_path(repository_root: Path, preferred: str) -> str:
    stem = preferred.removesuffix(".json")
    candidates = [preferred, *(f"{stem}.recovery-{index:03d}.json" for index in range(1, 101))]
    for candidate in candidates:
        path = contained_path(repository_root, candidate)
        if not path.exists() and not path.is_symlink():
            return candidate
    raise ContractError("paired job exhausted its immutable quarantine paths")


def _load_or_initialize_state(
    path: Path,
    plan_sha256: str,
    *,
    repository_root: Path,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema": JOB_STATE_SCHEMA,
            "revision": 0,
            "planSha256": plan_sha256,
            "status": "initialized",
            "attempts": [],
        }
    raw = load_json(path)
    root = require_mapping(raw, "paired job state")
    require_exact_keys(root, _STATE_KEYS, "paired job state")
    if root.get("schema") != JOB_STATE_SCHEMA:
        raise ContractError("unsupported paired job state schema")
    if root.get("planSha256") != plan_sha256:
        raise ContractError("resume state belongs to a different plan")
    require_int(root, "revision", "paired job state", minimum=1, maximum=1_000_000_000)
    require_enum(
        root,
        "status",
        "paired job state",
        {"initialized", "running", "interrupted", "completed", "completed_with_quarantine"},
    )
    attempts = require_list(root, "attempts", "paired job state", maximum_items=20_000)
    _validate_attempt_records(
        attempts,
        repository_root=repository_root,
        plan=plan,
        context="paired job state",
    )
    return dict(root)


def _replace_job_state_cas(
    path: Path,
    *,
    expected_revision: int,
    plan_sha256: str,
    status: str,
    attempts: list[Mapping[str, Any]],
) -> int:
    """Publish one state revision only if the durable predecessor is unchanged."""

    if path.exists() or path.is_symlink():
        current = require_mapping(load_json(path), "paired job state CAS predecessor")
        if current.get("schema") != JOB_STATE_SCHEMA:
            raise ContractError("paired job state CAS predecessor has the wrong schema")
        observed = require_int(
            current,
            "revision",
            "paired job state CAS predecessor",
            minimum=1,
            maximum=1_000_000_000,
        )
        if observed != expected_revision:
            raise ContractError("paired job state revision changed concurrently")
        if current.get("planSha256") != plan_sha256:
            raise ContractError("paired job state CAS predecessor belongs to another plan")
    elif expected_revision != 0:
        raise ContractError("paired job state disappeared during CAS publication")
    revision = expected_revision + 1
    replace_json_state(
        path,
        {
            "schema": JOB_STATE_SCHEMA,
            "revision": revision,
            "planSha256": plan_sha256,
            "status": status,
            "attempts": attempts,
        },
    )
    observed = require_mapping(load_json(path), "published paired job state")
    if observed.get("revision") != revision:
        raise ContractError("paired job state CAS publication was replaced")
    return revision


def _validate_attempt_records(
    attempts: list[Any],
    *,
    repository_root: Path,
    plan: Mapping[str, Any],
    context: str,
) -> list[Mapping[str, Any]]:
    jobs = require_list(plan, "jobs", "paired plan", maximum_items=5_000)
    jobs_by_id = {
        str(require_mapping(job, "paired plan job")["jobId"]): require_mapping(
            job, "paired plan job"
        )
        for job in jobs
    }
    job_ids = set(jobs_by_id)
    seen: set[tuple[str, int]] = set()
    attempt_numbers: dict[str, list[int]] = {}
    validated: list[Mapping[str, Any]] = []
    for index, raw_attempt in enumerate(attempts):
        attempt_context = f"{context}.attempts[{index}]"
        attempt = require_mapping(raw_attempt, attempt_context)
        require_exact_keys(attempt, _ATTEMPT_KEYS, attempt_context)
        job_id = require_identifier(attempt, "jobId", attempt_context)
        if job_id not in job_ids:
            raise ContractError(f"resume state contains an unknown job ID: {job_id}")
        number = require_int(attempt, "attempt", attempt_context, minimum=1, maximum=100)
        if (job_id, number) in seen:
            raise ContractError("duplicate job attempt in resume state")
        seen.add((job_id, number))
        attempt_numbers.setdefault(job_id, []).append(number)
        status = require_enum(
            attempt, "status", attempt_context, {"running", "completed", "quarantined"}
        )
        if status == "running":
            if (
                any(
                    attempt.get(key) is not None
                    for key in (
                        "returnCode",
                        "timedOut",
                        "outputLimitExceeded",
                        "memoryLimitExceeded",
                        "peakRssBytes",
                        "rssMeasurement",
                        "stdout",
                        "stderr",
                        "report",
                        "quarantine",
                        "failureCategory",
                        "commandReceipt",
                    )
                )
                or attempt.get("csa") != []
            ):
                raise ContractError(f"{attempt_context} running journal is not empty")
            validate_utc_timestamp(attempt.get("completedAt"), f"{attempt_context}.completedAt")
            validated.append(attempt)
            continue
        require_int(attempt, "returnCode", attempt_context, minimum=-255, maximum=255)
        timed_out = require_bool(attempt, "timedOut", attempt_context)
        output_limit = require_bool(attempt, "outputLimitExceeded", attempt_context)
        memory_limit = require_bool(attempt, "memoryLimitExceeded", attempt_context)
        peak_rss = attempt.get("peakRssBytes")
        if peak_rss is not None:
            require_int(attempt, "peakRssBytes", attempt_context, minimum=0, maximum=2**63 - 1)
        measurement = require_enum(
            attempt,
            "rssMeasurement",
            attempt_context,
            {
                "process_tree_ps_rss_sum",
                "process_tree_ps_short_lived_no_sample",
                "unavailable",
            },
        )
        stdout = ArtifactRef.from_dict(attempt.get("stdout"), f"{attempt_context}.stdout")
        stderr = ArtifactRef.from_dict(attempt.get("stderr"), f"{attempt_context}.stderr")
        verify_artifact_ref(repository_root, stdout)
        verify_artifact_ref(repository_root, stderr)
        failure = attempt.get("failureCategory")
        report_raw = attempt.get("report")
        csa_raw = attempt.get("csa")
        quarantine_raw = attempt.get("quarantine")
        if not isinstance(csa_raw, list) or len(csa_raw) > 2:
            raise ContractError(
                f"{attempt_context}.csa must contain at most two artifact references"
            )
        csa = [
            ArtifactRef.from_dict(value, f"{attempt_context}.csa[{csa_index}]")
            for csa_index, value in enumerate(csa_raw)
        ]
        for reference in csa:
            verify_artifact_ref(repository_root, reference)
        if status == "completed":
            if (
                failure is not None
                or timed_out
                or output_limit
                or memory_limit
                or attempt["returnCode"] != 0
            ):
                raise ContractError(f"{attempt_context} has inconsistent completed status")
            if quarantine_raw is not None:
                raise ContractError(f"{attempt_context} completed pair has quarantine evidence")
            report = ArtifactRef.from_dict(report_raw, f"{attempt_context}.report")
            verify_artifact_ref(repository_root, report)
            if len(csa) != 2:
                raise ContractError(f"{attempt_context} completed pair must have two CSA artifacts")
        else:
            if failure not in {
                "timeout",
                "output_limit",
                "memory_limit",
                "nonzero_exit",
                "missing_or_invalid_artifact",
            }:
                raise ContractError(f"{attempt_context} has an invalid failure category")
            if report_raw is not None or csa:
                raise ContractError(
                    f"{attempt_context} quarantined pair must not publish game artifacts"
                )
            quarantine = ArtifactRef.from_dict(quarantine_raw, f"{attempt_context}.quarantine")
            verify_artifact_ref(repository_root, quarantine)
            _validate_quarantine_record(
                load_json_artifact(repository_root, quarantine),
                attempt=attempt,
                job=jobs_by_id[job_id],
                repository_root=repository_root,
                context=f"{attempt_context}.quarantine",
            )
        _validate_rss_result(
            measurement=measurement,
            peak_rss_bytes=peak_rss,
            memory_limit_bytes=int(plan["memoryPerWorkerMiB"]) * 1024 * 1024,
            memory_limit_exceeded=memory_limit,
            completed=status == "completed",
            context=attempt_context,
        )
        validate_utc_timestamp(attempt.get("completedAt"), f"{attempt_context}.completedAt")
        command_receipt = ArtifactRef.from_dict(
            attempt.get("commandReceipt"), f"{attempt_context}.commandReceipt"
        )
        verify_artifact_ref(repository_root, command_receipt)
        evidence = _validate_attempt_command_receipt(
            load_json_artifact(repository_root, command_receipt),
            repository_root=repository_root,
            plan=plan,
            job=jobs_by_id[job_id],
            attempt_number=number,
            context=f"{attempt_context}.commandReceipt",
        )
        if _attempt_record_from_evidence(evidence, command_receipt) != attempt:
            raise ContractError(f"{attempt_context} differs from its command receipt evidence")
        validated.append(attempt)
    for job_id, numbers in attempt_numbers.items():
        ordered_numbers = sorted(numbers)
        if ordered_numbers != list(range(1, max(numbers) + 1)):
            raise ContractError(f"resume attempts are not contiguous for job {job_id}")
        rows = sorted(
            (row for row in validated if row.get("jobId") == job_id),
            key=lambda row: int(row["attempt"]),
        )
        if any(row.get("status") != "quarantined" for row in rows[:-1]):
            raise ContractError(
                f"only quarantined attempts may precede the final attempt for job {job_id}"
            )
        timestamps = [
            datetime.fromisoformat(str(row["completedAt"]).removesuffix("Z") + "+00:00")
            for row in rows
        ]
        if timestamps != sorted(timestamps):
            raise ContractError(f"attempt timestamps are out of order for job {job_id}")
    return validated


def _validate_quarantine_record(
    raw: object,
    *,
    attempt: Mapping[str, Any],
    job: Mapping[str, Any],
    repository_root: Path,
    context: str,
) -> None:
    quarantine = require_mapping(raw, context)
    require_exact_keys(
        quarantine,
        {
            "schema",
            "jobId",
            "attempt",
            "failureCategory",
            "returnCode",
            "timedOut",
            "outputLimitExceeded",
            "memoryLimitExceeded",
            "peakRssBytes",
            "rssMeasurement",
            "stdout",
            "stderr",
            "observedReport",
            "observedCsa",
            "failureDetail",
        },
        context,
    )
    if quarantine.get("schema") != "phase6_quarantined_job/v1":
        raise ContractError(f"{context} has an unsupported schema")
    for key in (
        "jobId",
        "attempt",
        "failureCategory",
        "returnCode",
        "timedOut",
        "outputLimitExceeded",
        "memoryLimitExceeded",
        "peakRssBytes",
        "rssMeasurement",
        "stdout",
        "stderr",
    ):
        if quarantine.get(key) != attempt.get(key):
            raise ContractError(f"{context}.{key} differs from its attempt record")
    observed_report_raw = quarantine.get("observedReport")
    observed_report = (
        None
        if observed_report_raw is None
        else ArtifactRef.from_dict(observed_report_raw, f"{context}.observedReport")
    )
    if observed_report is not None:
        verify_artifact_ref(repository_root, observed_report)
    observed_csa = quarantine.get("observedCsa")
    if not isinstance(observed_csa, list) or len(observed_csa) > 2:
        raise ContractError(f"{context}.observedCsa must contain at most two references")
    observed_csa_refs = [
        ArtifactRef.from_dict(reference, f"{context}.observedCsa[{index}]")
        for index, reference in enumerate(observed_csa)
    ]
    for reference in observed_csa_refs:
        verify_artifact_ref(repository_root, reference)
    if attempt.get("failureCategory") != "missing_or_invalid_artifact":
        if observed_report is not None or observed_csa_refs:
            raise ContractError(
                f"{context} non-artifact failure must not claim observed game artifacts"
            )
    elif (observed_report is None) != (len(observed_csa_refs) == 0):
        raise ContractError(f"{context} observed artifacts must be all present or all absent")
    elif observed_report is not None:
        if observed_report.path != job.get("reportPath") or [
            reference.path for reference in observed_csa_refs
        ] != job.get("csaPaths"):
            raise ContractError(f"{context} observed artifacts differ from planned paths")
        if len(observed_csa_refs) != 2:
            raise ContractError(f"{context} must bind exactly two observed CSA artifacts")
    failure_detail = quarantine.get("failureDetail")
    if failure_detail is not None and (
        not isinstance(failure_detail, str)
        or not failure_detail
        or len(failure_detail) > 1_024
        or "\x00" in failure_detail
    ):
        raise ContractError(f"{context}.failureDetail must be a bounded string or null")


def _latest_attempts(attempts: list[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    latest: dict[str, Mapping[str, Any]] = {}
    for attempt in attempts:
        job_id = str(attempt["jobId"])
        if job_id not in latest or int(attempt["attempt"]) > int(latest[job_id]["attempt"]):
            latest[job_id] = attempt
    return latest


def _command_argv(table: Mapping[str, Any], kind: str) -> list[str]:
    raw = require_list(
        table,
        "argv",
        "command",
        minimum_items=2,
        maximum_items=MAX_COMMAND_ARGUMENTS,
    )
    argv: list[str] = []
    total = 0
    for index, item in enumerate(raw):
        if not isinstance(item, str) or not item or "\x00" in item:
            raise ContractError(f"command.argv[{index}] must be a non-empty string")
        encoded_length = len(item.encode("utf-8"))
        if encoded_length > 4_096:
            raise ContractError(f"command.argv[{index}] is too large")
        total += encoded_length
        argv.append(item)
    if total > MAX_ARGUMENT_BYTES:
        raise ContractError("command argv exceeds its byte limit")
    if kind == "engine_arena" and "--resume" in argv:
        raise ContractError("resume is state-derived and must not be embedded in a plan")
    return argv


def _validate_model_spec(raw: object, context: str) -> Mapping[str, Any]:
    model = require_mapping(raw, context)
    require_exact_keys(model, {"modelId", "artifact", "evaluatorKind"}, context)
    require_identifier(model, "modelId", context)
    ArtifactRef.from_dict(model.get("artifact"), f"{context}.artifact")
    require_enum(
        model,
        "evaluatorKind",
        context,
        {"material", "handcrafted-baseline", "handcrafted-experimental", "neural"},
    )
    return model


def _validate_arena_argv(
    argv: list[str],
    *,
    job: Mapping[str, Any],
    model_a: Mapping[str, Any],
    model_b: Mapping[str, Any],
    git_commit: str,
    nodes_per_move: int,
    context: str,
) -> None:
    value_options = {
        "--games",
        "--player-a",
        "--player-b",
        "--a-depth",
        "--b-depth",
        "--a-hash-mb",
        "--b-hash-mb",
        "--a-model",
        "--b-model",
        "--nodes",
        "--max-plies",
        "--seed",
        "--sfen",
        "--git-commit",
        "--output-dir",
    }
    if len(argv) < 2 or argv[1] != "arena":
        raise ContractError(f"{context} command must invoke arena")
    values: dict[str, str] = {}
    index = 2
    while index < len(argv):
        option = argv[index]
        if option not in value_options or option in values or index + 1 >= len(argv):
            raise ContractError(f"{context} command contains an unknown or duplicate option")
        values[option] = argv[index + 1]
        index += 2
    required = value_options - {"--a-model", "--b-model"}
    if not required.issubset(values):
        raise ContractError(f"{context} command is missing a required arena option")
    if values["--games"] != "2":
        raise ContractError(f"{context} command must run exactly two paired games")
    if values["--nodes"] != str(nodes_per_move):
        raise ContractError(f"{context} command node limit differs from its plan")
    if values["--max-plies"] != str(PHASE6_MAX_PLIES):
        raise ContractError(f"{context} command max plies must be {PHASE6_MAX_PLIES}")
    if values["--git-commit"] != git_commit:
        raise ContractError(f"{context} command git commit differs from its plan")
    for left, right, minimum, maximum in (
        ("--a-depth", "--b-depth", 1, 64),
        ("--a-hash-mb", "--b-hash-mb", 1, 1_024),
    ):
        try:
            left_value = int(values[left])
            right_value = int(values[right])
        except ValueError as error:
            raise ContractError(f"{context} command contains a non-integer limit") from error
        if left_value != right_value or not minimum <= left_value <= maximum:
            raise ContractError(f"{context} command has inconsistent player limits")
    if values["--output-dir"] != job["outputDir"]:
        raise ContractError(f"{context} command output directory differs from its manifest")
    if values["--sfen"] != job["sfen"] or values["--seed"] != str(job["seed"]):
        raise ContractError(f"{context} command start or seed differs from its manifest")
    for side, model in (("a", model_a), ("b", model_b)):
        evaluator = str(model["evaluatorKind"])
        if values[f"--player-{side}"] != evaluator:
            raise ContractError(f"{context} command evaluator differs from its model spec")
        model_option = f"--{side}-model"
        artifact = ArtifactRef.from_dict(model.get("artifact"), f"{context}.model-{side}")
        if evaluator == "neural":
            if values.get(model_option) != artifact.path:
                raise ContractError(f"{context} neural command lacks its exact model artifact")
        elif model_option in values:
            raise ContractError(f"{context} non-neural command must not carry a model artifact")
    for option in ("--a-model", "--b-model", "--output-dir"):
        if option in values:
            validate_relative_path(values[option])


def _derive_job_seed(base_seed: int, kind: str, pair_index: int, start_id: str) -> int:
    digest = hashlib.sha256(
        f"phase6\0{base_seed}\0{kind}\0{pair_index}\0{start_id}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % 9_007_199_254_740_992


def _validate_plan_sfen(value: str, context: str) -> None:
    fields = value.split(" ")
    if (
        not value.isascii()
        or len(fields) != 4
        or any(not field for field in fields)
        or fields[1] not in {"b", "w"}
        or fields[3] != "1"
        or any(character in value for character in "\r\n")
    ):
        raise ContractError(f"{context}.sfen must be one canonical four-field SFEN")


def _parse_closed_options(
    argv: list[str],
    *,
    prefix: tuple[str, ...],
    required: set[str],
    optional: set[str] | None = None,
    flags: set[str] | None = None,
    context: str,
) -> tuple[dict[str, str], set[str]]:
    """Parse one exact option/value command without accepting positional drift."""

    if tuple(argv[: len(prefix)]) != prefix:
        raise ContractError(f"{context} must invoke the exact repository command")
    optional = optional or set()
    flags = flags or set()
    allowed = required | optional | flags
    values: dict[str, str] = {}
    observed_flags: set[str] = set()
    index = len(prefix)
    while index < len(argv):
        option = argv[index]
        if option not in allowed or option in values or option in observed_flags:
            raise ContractError(f"{context} contains an unknown or duplicate option: {option}")
        if option in flags:
            observed_flags.add(option)
            index += 1
            continue
        if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
            raise ContractError(f"{context} option lacks a value: {option}")
        values[option] = argv[index + 1]
        index += 2
    missing = sorted(required - set(values))
    if missing:
        raise ContractError(f"{context} is missing required options: {missing}")
    return values, observed_flags


def _validate_runtime_arena_command(argv: list[str]) -> None:
    required = {
        "--games",
        "--player-a",
        "--player-b",
        "--a-depth",
        "--b-depth",
        "--a-hash-mb",
        "--b-hash-mb",
        "--nodes",
        "--max-plies",
        "--seed",
        "--sfen",
        "--git-commit",
        "--output-dir",
    }
    values, _ = _parse_closed_options(
        argv,
        prefix=(argv[0], "arena"),
        required=required,
        optional={"--a-model", "--b-model"},
        flags={"--resume"},
        context="engine arena command",
    )
    if values["--games"] != "2":
        raise ContractError("engine arena command must run exactly two games")
    for option, minimum, maximum in (
        ("--a-depth", 1, 64),
        ("--b-depth", 1, 64),
        ("--a-hash-mb", 1, 1_024),
        ("--b-hash-mb", 1, 1_024),
        ("--nodes", 1, 1_000_000_000),
        ("--max-plies", 1, 10_000),
        ("--seed", 0, 9_007_199_254_740_991),
    ):
        _bounded_decimal(values[option], option, minimum=minimum, maximum=maximum)
    _validate_plan_sfen(values["--sfen"], "engine arena command")


def _validate_engine_validation_command(argv: list[str]) -> None:
    values, _ = _parse_closed_options(
        argv,
        prefix=(argv[0], "perft"),
        required={"--depth", "--sfen"},
        context="engine validation command",
    )
    if values["--depth"] != "0":
        raise ContractError("engine validation command depth must be zero")
    _validate_plan_sfen(values["--sfen"], "engine validation command")


def _validate_dataset_export_command(argv: list[str]) -> None:
    values, _ = _parse_closed_options(
        argv,
        prefix=(argv[0], "export-csa-jsonl"),
        required={"--input-dir", "--output", "--max-games"},
        context="engine dataset-export command",
    )
    _bounded_decimal(values["--max-games"], "--max-games", minimum=1, maximum=100)


def _validate_model_train_command(argv: list[str]) -> None:
    if argv[:4] != ["python", "-m", "open_shogi_training.models", "train"]:
        raise ContractError("model command must invoke the exact repository model module")
    _parse_closed_options(
        argv,
        prefix=("python", "-m", "open_shogi_training.models", "train"),
        required={
            "--features",
            "--model",
            "--training",
            "--labels",
            "--positions",
            "--dataset-manifest",
            "--replay-manifest",
            "--output-dir",
        },
        optional={"--resume"},
        context="model training command",
    )


def _validate_labeling_command(argv: list[str]) -> None:
    values, _ = _parse_closed_options(
        argv,
        prefix=("python", "-m", "open_shogi_training.labeling", "label"),
        required={
            "--config",
            "--project-root",
            "--positions",
            "--dataset-manifest",
            "--benchmark-report",
            "--output-dir",
            "--target-completed",
        },
        context="teacher labeling command",
    )
    if values["--project-root"] != ".":
        raise ContractError("labeling command project root must be the repository cwd")
    if _bounded_decimal(
        values["--target-completed"],
        "--target-completed",
        minimum=1,
        maximum=10_000,
    ) not in {10, 100, 1_000, 10_000}:
        raise ContractError("labeling target must be an approved cumulative milestone")


def _validate_contained_command_paths(
    repository_root: Path,
    argv: list[str],
    *,
    input_files: set[str] | None = None,
    input_directories: set[str] | None = None,
    output_paths: set[str] | None = None,
) -> None:
    input_files = input_files or set()
    input_directories = input_directories or set()
    output_paths = output_paths or set()
    path_options = input_files | input_directories | output_paths
    for index, option in enumerate(argv):
        if option not in path_options:
            continue
        if index + 1 >= len(argv):
            raise ContractError(f"command path option lacks a value: {option}")
        relative = validate_relative_path(argv[index + 1])
        candidate = contained_path(repository_root, relative)
        if option in input_files and (
            not candidate.exists() or candidate.is_symlink() or not candidate.is_file()
        ):
            raise ContractError(f"command input must be a regular non-symlink file: {relative}")
        if option in input_directories and (
            not candidate.exists() or candidate.is_symlink() or not candidate.is_dir()
        ):
            raise ContractError(f"command input must be a regular directory: {relative}")


def _bounded_decimal(value: str, option: str, *, minimum: int, maximum: int) -> int:
    if not value.isascii() or not value.isdecimal():
        raise ContractError(f"command {option} must be an unsigned decimal integer")
    parsed = int(value)
    if not minimum <= parsed <= maximum:
        raise ContractError(f"command {option} is outside {minimum}..{maximum}")
    return parsed


def _snapshot_python_runtime(
    repository_root: Path, *, supplied_commit: str | None
) -> _PythonSourceSnapshot:
    """Seal the tracked Python package so later worktree edits cannot be imported."""

    commit = require_clean_head(repository_root, supplied_commit)
    try:
        completed = subprocess.run(
            [
                "/usr/bin/git",
                "ls-tree",
                "-rz",
                commit,
                "--",
                "training/open_shogi_training",
            ],
            cwd=repository_root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
            env=_sanitized_environment(),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ContractError("cannot enumerate the immutable Python runtime") from error
    if completed.returncode != 0 or len(completed.stdout) > 1024 * 1024:
        raise ContractError("cannot enumerate the immutable Python runtime")
    raw_entries = completed.stdout.split(b"\0")
    files: list[tuple[str, str, int]] = []
    payloads: dict[str, bytes] = {}
    total = 0
    for raw_entry in raw_entries:
        if not raw_entry:
            continue
        try:
            metadata, raw_path = raw_entry.split(b"\t", 1)
            mode, kind, object_id = metadata.split(b" ", 2)
            relative = raw_path.decode("utf-8", errors="strict")
            object_id_text = object_id.decode("ascii", errors="strict")
        except (ValueError, UnicodeDecodeError) as error:
            raise ContractError("Python runtime contains a malformed Git tree entry") from error
        if kind != b"blob" or mode not in {b"100644", b"100755"}:
            raise ContractError("Python runtime inputs must be regular Git blobs")
        if not re.fullmatch(r"[0-9a-f]{40,64}", object_id_text):
            raise ContractError("Python runtime Git blob has an invalid object identity")
        if not relative.startswith("training/open_shogi_training/") or not relative.endswith(".py"):
            continue
        if len(files) >= 2_000:
            raise ContractError("Python runtime contains too many source files")
        validate_relative_path(relative)
        payload = _read_git_blob(repository_root, object_id_text, maximum_bytes=4 * 1024 * 1024)
        total += len(payload)
        if total > 128 * 1024 * 1024:
            raise ContractError("Python runtime exceeds its aggregate byte bound")
        digest = hashlib.sha256(payload).hexdigest()
        files.append((relative, digest, 4 * 1024 * 1024))
        payloads[relative] = payload
    if not files or not any(path.endswith("/__init__.py") for path, _, _ in files):
        raise ContractError("tracked Python runtime package is incomplete")
    source_digest = hashlib.sha256(b"phase6_python_source_snapshot/v1\x00")
    for relative, sha256, _ in sorted(files):
        payload = payloads[relative]
        encoded = relative.encode("utf-8")
        source_digest.update(len(encoded).to_bytes(4, "big"))
        source_digest.update(encoded)
        source_digest.update(len(payload).to_bytes(8, "big"))
        source_digest.update(bytes.fromhex(sha256))
    source_identity: dict[str, object] = {
        "treeSha256": source_digest.hexdigest(),
        "files": len(files),
        "bytes": total,
    }
    require_clean_head(repository_root, commit)
    storage = ensure_contained_directory(repository_root, "local/runtime-snapshots")
    runtime = RuntimeTreeSnapshot.create(
        project_root=repository_root,
        working_directory="training",
        files=tuple(files),
        storage_directory=storage,
        payloads=payloads,
    )
    try:
        require_clean_head(repository_root, commit)
    except BaseException:
        runtime.close()
        raise
    return _PythonSourceSnapshot(runtime=runtime, identity=source_identity)


def _read_git_blob(repository_root: Path, object_id: str, *, maximum_bytes: int) -> bytes:
    try:
        completed = subprocess.run(
            ["/usr/bin/git", "cat-file", "blob", object_id],
            cwd=repository_root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
            env=_sanitized_environment(),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ContractError("cannot read an immutable Python runtime Git blob") from error
    if completed.returncode != 0 or len(completed.stdout) > maximum_bytes:
        raise ContractError("Python runtime Git blob exceeds its bound or is unavailable")
    return completed.stdout


def _isolated_python_command(
    interpreter: str,
    runtime_training: Path,
    site_paths: tuple[str, ...],
    argv: list[str],
) -> list[str]:
    if len(argv) < 4 or argv[:2] != ["python", "-m"]:
        raise ContractError("Python pipeline command has no exact module boundary")
    module = argv[2]
    if len(site_paths) != 1:
        raise ContractError("Python runtime must use one receipt-bound site-packages tree")
    launcher = (
        "import runpy,sys;"
        "root=sys.argv.pop(1);n=int(sys.argv.pop(1));"
        "paths=[sys.argv.pop(1) for _ in range(n)];"
        "module=sys.argv.pop(1);sys.path[:0]=[root,*paths];"
        "sys.argv=[module,*sys.argv[1:]];"
        "runpy.run_module(module,run_name='__main__',alter_sys=True)"
    )
    return [
        interpreter,
        "-I",
        "-S",
        "-c",
        launcher,
        str(runtime_training),
        str(len(site_paths)),
        *site_paths,
        module,
        *argv[3:],
    ]


def _sanitized_environment() -> dict[str, str]:
    environment: dict[str, str] = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "LC_ALL": "C",
        "LANG": "C",
    }
    for key in ("TMPDIR", "TMP", "TEMP"):
        value = os.environ.get(key)
        if value:
            environment[key] = value
    return environment


def _bind_process_start_identity(process: subprocess.Popen[bytes]) -> str | None:
    """Bind the owned leader while it is live, tolerating a transient Darwin gap."""

    deadline = time.monotonic() + _PROCESS_IDENTITY_BIND_TIMEOUT_SECONDS
    while True:
        identity = _process_start_identity(process.pid)
        if identity is not None:
            return identity
        if process.poll() is not None:
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        time.sleep(min(_PROCESS_IDENTITY_BIND_RETRY_SECONDS, remaining))


def _run_bounded_process(
    argv: list[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    pass_fds: tuple[int, ...] = (),
    memory_limit_bytes: int,
    environment: Mapping[str, str] | None = None,
    launch_guard: Callable[[], object] | None = None,
) -> tuple[bytes, bytes, int, bool, bool, bool, int | None, str]:
    process_environment = _sanitized_environment() if environment is None else dict(environment)
    if launch_guard is not None:
        launch_guard()
    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=process_environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            start_new_session=True,
            pass_fds=pass_fds,
        )
    except OSError as error:
        raise ContractError(f"cannot start bounded repository command: {error}") from error
    try:
        if launch_guard is not None:
            launch_guard()
    except BaseException:
        # The child may have started before the post-spawn identity check failed.
        # Stop the new session immediately and close every inherited pipe before
        # propagating the original snapshot error.  The executable snapshot is a
        # private immutable pathname; active same-UID flag clearing remains an
        # explicit platform threat-model exclusion rather than an exact-FD claim.
        identities: dict[int, str] = {}
        leader_identity = _process_start_identity(process.pid)
        if leader_identity is None:
            # The unreaped Popen child itself is still an owned handle even when
            # the process table cannot establish a safe group identity.
            with suppress(ProcessLookupError):
                process.kill()
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=2)
        else:
            identities[process.pid] = leader_identity
            try:
                _, live_processes = _process_tree_rss_bytes(process.pid)
                identities.update(getattr(live_processes, "identities", {}))
            except ContractError:
                live_processes = _ObservedProcessIds({process.pid}, identities)
            _terminate_process_tree(process, live_processes | {process.pid}, identities)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                with suppress(OSError):
                    stream.close()
        raise
    assert process.stdout is not None
    assert process.stderr is not None
    overflow = threading.Event()
    reader_shutdown = threading.Event()
    stdout = bytearray()
    stderr = bytearray()
    readers = [
        threading.Thread(
            target=_read_pipe,
            args=(process.stdout, stdout, overflow, reader_shutdown),
            daemon=True,
        ),
        threading.Thread(
            target=_read_pipe,
            args=(process.stderr, stderr, overflow, reader_shutdown),
            daemon=True,
        ),
    ]
    for reader in readers:
        reader.start()
    leader_start_identity = _bind_process_start_identity(process)
    if leader_start_identity is None and process.poll() is None:
        # The process table is also our PID-reuse authority.  Continuing without
        # a launch identity would make every later group signal ambiguous.
        with suppress(ProcessLookupError):
            process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                process.kill()
            process.wait(timeout=2)
        reader_shutdown.set()
        for stream in (process.stdout, process.stderr):
            with suppress(OSError):
                stream.close()
        for reader in readers:
            reader.join(timeout=1)
        raise ContractError("cannot bind repository command process-start identity")
    deadline = time.monotonic() + timeout_seconds
    timed_out = False
    memory_limit_exceeded = False
    peak_rss_bytes: int | None = None
    rss_measurement = "process_tree_ps_rss_sum"
    measurement_failed = False
    leader_finished_at: float | None = None
    last_live_processes: set[int] = {process.pid}
    process_identities: dict[int, str] = (
        {process.pid: leader_start_identity} if leader_start_identity is not None else {}
    )
    while True:
        try:
            observed_rss, live_processes = _process_tree_rss_bytes(process.pid)
            process_identities.update(getattr(live_processes, "identities", {}))
            if live_processes:
                last_live_processes = live_processes
        except ContractError:
            observed_rss = None
            if process.poll() is not None:
                # A process may exit between the exact ps snapshot and the
                # per-PID identity lookup. Once the owned leader is reaped there
                # is no unmetered live process; classify a never-sampled run as
                # the explicit short-lived shape below.
                live_processes = _ObservedProcessIds(set(), {})
            else:
                measurement_failed = True
                live_processes = last_live_processes
                # A memory limit that cannot be measured is not a limit.  Stop at
                # the first failed live sample rather than running unmetered.
                _terminate_process_tree(process, live_processes, process_identities)
                break
        owned_processes = live_processes | last_live_processes
        if observed_rss is not None:
            peak_rss_bytes = max(peak_rss_bytes or 0, observed_rss)
            if observed_rss > memory_limit_bytes:
                memory_limit_exceeded = True
                _terminate_process_tree(process, owned_processes, process_identities)
                break
        elif not live_processes and process.poll() is None:
            # A missing process-table row is benign only after the leader has
            # already exited in this exact sample. Otherwise we have no evidence
            # that the configured memory ceiling is being enforced.
            measurement_failed = True
            _terminate_process_tree(process, owned_processes, process_identities)
            break
        if overflow.is_set():
            _terminate_process_tree(process, owned_processes, process_identities)
            break
        if time.monotonic() >= deadline:
            timed_out = True
            _terminate_process_tree(process, owned_processes, process_identities)
            break
        if process.poll() is not None:
            if leader_finished_at is None:
                leader_finished_at = time.monotonic()
            if live_processes:
                _terminate_process_tree(process, owned_processes, process_identities)
            if not any(reader.is_alive() for reader in readers):
                break
            if time.monotonic() - leader_finished_at >= 2:
                _kill_process_tree(process, owned_processes, process_identities)
                break
        overflow.wait(0.05)
    try:
        return_code = process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _kill_process_tree(process, last_live_processes, process_identities)
        return_code = process.wait(timeout=5)
    for reader in readers:
        reader.join(timeout=5)
    if any(reader.is_alive() for reader in readers):
        _kill_process_tree(process, last_live_processes, process_identities)
        reader_shutdown.set()
        for stream in (process.stdout, process.stderr):
            with suppress(OSError):
                stream.close()
        for reader in readers:
            reader.join(timeout=1)
        if any(reader.is_alive() for reader in readers):
            overflow.set()
    if measurement_failed:
        rss_measurement = "unavailable"
        memory_limit_exceeded = True
        peak_rss_bytes = None
    elif peak_rss_bytes is None:
        peak_rss_bytes = 0
        rss_measurement = "process_tree_ps_short_lived_no_sample"
    return (
        bytes(stdout),
        bytes(stderr),
        return_code,
        timed_out,
        overflow.is_set(),
        memory_limit_exceeded,
        peak_rss_bytes,
        rss_measurement,
    )


def _process_tree_rss_bytes(leader_pid: int) -> tuple[int | None, set[int]]:
    try:
        completed = subprocess.run(
            ["/bin/ps", "-axo", "pid=,ppid=,pgid=,rss="],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=2,
            env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "LC_ALL": "C", "LANG": "C"},
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ContractError("cannot measure command process-tree RSS") from error
    if completed.returncode != 0 or len(completed.stdout) > 8 * 1024 * 1024:
        raise ContractError("cannot measure command process-tree RSS")
    processes: dict[int, tuple[int, int, int]] = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) != 4:
            raise ContractError("command process-tree row is malformed")
        try:
            pid, ppid, pgid, rss_kib = (int(field) for field in fields[:4])
        except ValueError:
            raise ContractError("command process-tree row is malformed") from None
        if pid <= 0 or ppid < 0 or pgid < 0 or rss_kib < 0 or pid in processes:
            raise ContractError("command process-tree row is invalid")
        processes[pid] = (ppid, pgid, rss_kib)
    selected = {pid for pid, (_, pgid, _) in processes.items() if pgid == leader_pid}
    if leader_pid in processes:
        selected.add(leader_pid)
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _, _) in processes.items():
            if ppid in selected and pid not in selected:
                selected.add(pid)
                changed = True
    if not selected:
        return None, _ObservedProcessIds(set(), {})
    identities: dict[int, str] = {}
    live_selected: set[int] = set()
    for pid in selected:
        identity = _process_start_identity(pid)
        if identity is None:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                # A descendant may exit between the atomic `ps` snapshot and
                # the per-PID identity query. Its sampled RSS remains part of
                # this conservative total, but its numeric PID must never be
                # retained for a later signal.
                continue
            except PermissionError as error:
                raise ContractError("command process identity became unavailable") from error
            if _process_is_defunct_or_reused(pid, expected_process_group=processes[pid][1]):
                # A sampled descendant can become a zombie before libproc's
                # two-snapshot identity read.  Its sampled RSS remains in the
                # conservative total, but a defunct or reused numeric PID must
                # not be retained for later signalling.
                continue
            raise ContractError("command process identity became unavailable")
        identities[pid] = identity
        live_selected.add(pid)
    return (
        sum(processes[pid][2] for pid in selected) * 1024,
        _ObservedProcessIds(live_selected, identities),
    )


def _process_is_defunct_or_reused(pid: int, *, expected_process_group: int) -> bool:
    """Resolve the narrow ps-to-identity exit race without trusting a numeric PID."""

    try:
        completed = subprocess.run(
            ["/bin/ps", "-p", str(pid), "-o", "pid=,pgid=,state="],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=2,
            env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "LC_ALL": "C", "LANG": "C"},
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ContractError("cannot recheck command process identity") from error
    if completed.returncode not in {0, 1} or len(completed.stdout) > 4_096:
        raise ContractError("cannot recheck command process identity")
    rows = [line.split() for line in completed.stdout.splitlines() if line.strip()]
    if not rows:
        return True
    if len(rows) != 1 or len(rows[0]) != 3:
        raise ContractError("command process identity recheck is malformed")
    try:
        observed_pid = int(rows[0][0])
        observed_process_group = int(rows[0][1])
    except ValueError as error:
        raise ContractError("command process identity recheck is malformed") from error
    if observed_pid != pid or observed_process_group <= 0:
        raise ContractError("command process identity recheck is invalid")
    if observed_process_group != expected_process_group:
        return True
    return rows[0][2].startswith(b"Z")


def _read_pipe(
    stream: Any,
    output: bytearray,
    overflow: threading.Event,
    shutdown: threading.Event,
) -> None:
    try:
        while chunk := stream.read(64 * 1024):
            remaining = MAX_COMMAND_OUTPUT_BYTES - len(output)
            if len(chunk) > remaining:
                output.extend(chunk[: max(0, remaining)])
                overflow.set()
                return
            output.extend(chunk)
    except (OSError, ValueError):
        if not shutdown.is_set():
            # A spontaneous pipe-reader failure invalidates the bounded output
            # evidence.  Coordinator-requested closure after process-tree kill
            # is the sole benign close race.
            overflow.set()
    finally:
        with suppress(OSError, ValueError):
            stream.close()


def _terminate_process_tree(
    process: subprocess.Popen[bytes],
    process_ids: set[int],
    identities: Mapping[int, str] | None = None,
) -> None:
    if process.poll() is None:
        _signal_owned_process_group(process, signal.SIGTERM, identities or {})
    _signal_process_ids(
        process_ids,
        signal.SIGTERM,
        exclude={process.pid, os.getpid()},
        identities=identities or {},
    )
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        _kill_process_tree(process, process_ids, identities)


def _kill_process_tree(
    process: subprocess.Popen[bytes],
    process_ids: set[int],
    identities: Mapping[int, str] | None = None,
) -> None:
    if process.poll() is None:
        _signal_owned_process_group(process, signal.SIGKILL, identities or {})
    _signal_process_ids(
        process_ids,
        signal.SIGKILL,
        exclude={process.pid, os.getpid()},
        identities=identities or {},
    )


def _signal_owned_process_group(
    process: subprocess.Popen[bytes],
    signal_number: int,
    identities: Mapping[int, str],
) -> bool:
    """Signal the launched session only while its leader still has our identity.

    A numeric PGID can be recycled after its leader exits.  We therefore never
    issue ``killpg`` without re-reading the leader's process-start identity and
    confirming that it remains the session leader.  Escaped descendants are
    handled separately by ``_signal_process_ids`` with the same identity rule.
    """

    expected = identities.get(process.pid)
    if expected is None or _process_start_identity(process.pid) != expected:
        return False
    try:
        if os.getpgid(process.pid) != process.pid:
            return False
        os.killpg(process.pid, signal_number)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _signal_process_ids(
    process_ids: set[int],
    signal_number: int,
    *,
    exclude: set[int],
    identities: Mapping[int, str],
) -> None:
    for process_id in sorted(process_ids - exclude):
        if process_id <= 1:
            continue
        expected = identities.get(process_id)
        if expected is None or _process_start_identity(process_id) != expected:
            continue
        with suppress(ProcessLookupError, PermissionError):
            os.kill(process_id, signal_number)


def _process_start_identity(process_id: int) -> str | None:
    return read_process_identity(process_id)


def _validate_string_array(
    table: Mapping[str, Any],
    key: str,
    context: str,
    *,
    expected: int,
    identifiers: bool = False,
    paths: bool = False,
) -> tuple[str, ...]:
    rows = require_list(table, key, context, minimum_items=expected, maximum_items=expected)
    result: list[str] = []
    for index, value in enumerate(rows):
        if not isinstance(value, str):
            raise ContractError(f"{context}.{key}[{index}] must be a string")
        if identifiers:
            temporary = {key: value}
            require_identifier(temporary, key, f"{context}.{key}[{index}]")
        elif paths:
            validate_relative_path(value)
        result.append(value)
    return tuple(result)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
