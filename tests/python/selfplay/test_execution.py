from __future__ import annotations

import copy
import hashlib
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from open_shogi_training.selfplay import execution as execution_module
from open_shogi_training.selfplay.common import (
    ArtifactRef,
    ContractError,
    artifact_ref,
    canonical_json_bytes,
    canonical_sha256,
    load_json,
)
from open_shogi_training.selfplay.config import SelfPlayConfig
from open_shogi_training.selfplay.execution import (
    CommandOutcome,
    CommandRunner,
    _sanitized_environment,
    _snapshot_python_runtime,
    execute_paired_plan,
)
from open_shogi_training.selfplay.planning import (
    ModelSpec,
    build_selfplay_plan,
)

from .conftest import (
    held_directory_authority,
    make_phase2_pair_report,
    tiny_model_bytes,
    write_initial_registry_ref,
    write_ref,
    write_selfplay_config_ref,
    write_validated_phase3_starts,
)

SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 1"
VALIDATION_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/6P2/PPPPPP1PP/1B5R1/LNSGKGSNL w - 1"


class FakeRunner:
    def __init__(
        self,
        root: Path,
        *,
        fail_job: str | None = None,
        raise_job: str | None = None,
        corrupt_job: str | None = None,
    ) -> None:
        self.root = root
        self.fail_job = fail_job
        self.raise_job = raise_job
        self.corrupt_job = corrupt_job
        self.calls: list[str] = []

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
        del engine_build_receipt, resume, memory_limit_mib, receipt_path
        assert expected_executable is not None
        argv = command["argv"]
        output_dir = argv[argv.index("--output-dir") + 1]
        job_id = output_dir.rsplit("/", 1)[-1]
        self.calls.append(job_id)
        stdout = write_ref(self.root, stdout_path, b"ok\n")
        stderr = write_ref(self.root, stderr_path, b"")
        if job_id == self.raise_job:
            raise ContractError("simulated runner interruption")
        if job_id == self.fail_job:
            return CommandOutcome(
                2, False, False, False, 1, "process_tree_ps_rss_sum", stdout, stderr
            )
        csa_refs = (
            write_ref(self.root, f"{output_dir}/games/game-000001.csa", b"V3.0\n"),
            write_ref(self.root, f"{output_dir}/games/game-000002.csa", b"V3.0\n"),
        )
        model_path = argv[argv.index("--a-model") + 1]
        model_ref = artifact_ref(self.root, model_path)
        model_bytes = (self.root / model_path).read_bytes()
        report = make_phase2_pair_report(
            job={
                "jobId": job_id,
                "seed": int(argv[argv.index("--seed") + 1]),
                "sfen": argv[argv.index("--sfen") + 1],
            },
            model_a_ref=model_ref,
            model_a_bytes=model_bytes,
            model_b_ref=model_ref,
            model_b_bytes=model_bytes,
            csa_refs=csa_refs,
        )
        if job_id == self.corrupt_job:
            report["games"][0]["csaSha256"] = "f" * 64
        write_ref(
            self.root,
            f"{output_dir}/arena-report.json",
            canonical_json_bytes(report),
        )
        return CommandOutcome(0, False, False, False, 1, "process_tree_ps_rss_sum", stdout, stderr)


def _plan(tmp_path: Path, config: SelfPlayConfig) -> tuple[dict[str, object], ArtifactRef]:
    engine_ref = write_ref(tmp_path, "target/release/open-shogi-cli")
    starts, start_ref, validation_ref, _, _ = write_validated_phase3_starts(
        tmp_path,
        engine_ref=engine_ref,
        train_sfen=SFEN,
        validation_sfen=VALIDATION_SFEN,
    )
    model_ref = write_ref(tmp_path, "weights/champion.osaval", tiny_model_bytes())
    registry_ref = write_initial_registry_ref(
        tmp_path,
        champion_ref=model_ref,
        quantization="float32",
    )
    config_ref = write_selfplay_config_ref(tmp_path)
    plan = build_selfplay_plan(
        generation_id="generation-0001",
        champion=ModelSpec("champion-v0", model_ref, "neural"),
        engine=engine_ref,
        model_registry=registry_ref,
        git_commit="a399407",
        config=config,
        config_ref=config_ref,
        start_positions=starts,
        start_positions_ref=start_ref,
        validation_ref=validation_ref,
    )
    contents = canonical_json_bytes(plan)
    reference = write_ref(tmp_path, "artifacts/phase6/selfplay-plan.json", contents)
    return plan, reference


def test_paired_execution_resumes_without_replaying_completed_jobs(
    tmp_path: Path, selfplay_config: SelfPlayConfig
) -> None:
    plan, plan_ref = _plan(tmp_path, selfplay_config)
    runner = FakeRunner(tmp_path)
    arguments = {
        "repository_root": tmp_path,
        "plan": plan,
        "plan_ref": plan_ref,
        "state_path": "artifacts/phase6/state.json",
        "manifest_path": "artifacts/phase6/manifest.json",
        "runner": runner,
        "now": lambda: "2026-08-08T00:00:02Z",
    }

    first = execute_paired_plan(**arguments)
    state_path = tmp_path / "artifacts/phase6/state.json"
    state_before_resume = state_path.read_bytes()
    second = execute_paired_plan(**arguments)

    assert first == second
    assert state_path.read_bytes() == state_before_resume
    assert first["status"] == "completed"
    assert first["jobsCompleted"] == 20
    assert len(runner.calls) == 20


def test_paired_execution_rejects_a_second_process_holding_its_job_authority(
    tmp_path: Path,
    selfplay_config: SelfPlayConfig,
) -> None:
    plan, plan_ref = _plan(tmp_path, selfplay_config)
    runner = FakeRunner(tmp_path)
    with held_directory_authority(tmp_path / "artifacts/phase6"):
        with pytest.raises(ContractError, match="another process owns"):
            execute_paired_plan(
                repository_root=tmp_path,
                plan=plan,
                plan_ref=plan_ref,
                state_path="artifacts/phase6/state.json",
                manifest_path="artifacts/phase6/manifest.json",
                runner=runner,
            )
        assert runner.calls == []
        assert not (tmp_path / "artifacts/phase6/state.json").exists()


def test_python_runtime_uses_exact_git_blobs_and_rejects_sitecustomize(
    tmp_path: Path,
) -> None:
    package = tmp_path / "training/open_shogi_training"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 'committed'\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("/local/\n", encoding="utf-8")
    environment = {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"}
    for arguments in (
        ["init", "-q"],
        ["config", "user.name", "Fixture"],
        ["config", "user.email", "fixture@example.invalid"],
        ["add", ".gitignore", "training/open_shogi_training/__init__.py"],
        ["commit", "-qm", "fixture"],
    ):
        subprocess.run(
            ["/usr/bin/git", *arguments],
            cwd=tmp_path,
            env=environment,
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    commit = subprocess.run(
        ["/usr/bin/git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    runtime = _snapshot_python_runtime(tmp_path, supplied_commit=commit)
    try:
        assert (runtime.cwd / "open_shogi_training/__init__.py").read_text(
            encoding="utf-8"
        ) == "VALUE = 'committed'\n"
        (package / "__init__.py").write_text("VALUE = 'edited'\n", encoding="utf-8")
        runtime.assert_unchanged()
        assert (runtime.cwd / "open_shogi_training/__init__.py").read_text(
            encoding="utf-8"
        ) == "VALUE = 'committed'\n"
    finally:
        runtime.close()

    subprocess.run(
        ["/usr/bin/git", "restore", "training/open_shogi_training/__init__.py"],
        cwd=tmp_path,
        env=environment,
        check=True,
    )
    (tmp_path / "training/sitecustomize.py").write_text(
        "raise RuntimeError('must never import')\n", encoding="utf-8"
    )
    with pytest.raises(ContractError, match="untracked"):
        _snapshot_python_runtime(tmp_path, supplied_commit=commit)
    sanitized = _sanitized_environment()
    assert "PYTHONPATH" not in sanitized
    assert sanitized["PYTHONNOUSERSITE"] == "1"


def test_python_command_receipt_rejects_dependency_drift_on_resume(
    tmp_path: Path,
) -> None:
    package = tmp_path / "training/open_shogi_training/models"
    package.mkdir(parents=True)
    (tmp_path / "training/open_shogi_training/__init__.py").write_bytes(b"")
    (package / "__init__.py").write_bytes(b"")
    (package / "__main__.py").write_text("print('runtime-receipt-ok')\n", encoding="utf-8")
    inputs = {
        "features": "inputs/features.toml",
        "model": "inputs/model.toml",
        "training": "inputs/training.toml",
        "labels": "inputs/labels.jsonl",
        "positions": "inputs/positions.jsonl.gz",
        "dataset-manifest": "inputs/dataset.json",
        "replay-manifest": "inputs/replay.json",
    }
    for relative in inputs.values():
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fixture\n")
    (tmp_path / "uv.lock").write_bytes(b"version = 1\n")
    (tmp_path / ".gitignore").write_text("/artifacts/\n/local/\n", encoding="utf-8")
    environment = {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "LANG": "C",
        "GIT_AUTHOR_NAME": "Fixture",
        "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
        "GIT_COMMITTER_NAME": "Fixture",
        "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
    }
    for arguments in (
        ("init", "-q"),
        ("add", "."),
        ("commit", "-qm", "fixture runtime"),
    ):
        subprocess.run(
            ["/usr/bin/git", *arguments],
            cwd=tmp_path,
            env=environment,
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    argv = ["python", "-m", "open_shogi_training.models", "train"]
    for option, relative in inputs.items():
        argv.extend((f"--{option}", relative))
    argv.extend(("--output-dir", "artifacts/model-output"))
    command = {"kind": "model_cli", "argv": argv, "timeoutSeconds": 30}
    runner = CommandRunner(tmp_path, require_clean_repository=True)
    arguments = {
        "stdout_path": "artifacts/model.stdout",
        "stderr_path": "artifacts/model.stderr",
        "receipt_path": "artifacts/model.receipt.json",
        "memory_limit_mib": 1_024,
    }

    first = runner.run(command, **arguments)
    assert first.return_code == 0
    assert first.command_receipt is not None
    assert first.runtime_receipt is not None
    assert (tmp_path / first.stdout.path).read_bytes() == b"runtime-receipt-ok\n"
    assert runner.run(command, **arguments) == first

    (tmp_path / "uv.lock").write_bytes(b"version = 2\n")
    for arguments_ in (("add", "uv.lock"), ("commit", "-qm", "dependency drift")):
        subprocess.run(
            ["/usr/bin/git", *arguments_],
            cwd=tmp_path,
            env=environment,
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    with pytest.raises(ContractError, match="different Python runtime"):
        runner.run(command, **arguments)


def test_failed_pair_is_quarantined_without_losing_other_results(
    tmp_path: Path, selfplay_config: SelfPlayConfig
) -> None:
    plan, plan_ref = _plan(tmp_path, selfplay_config)
    runner = FakeRunner(tmp_path, fail_job="pair-0003")

    manifest = execute_paired_plan(
        repository_root=tmp_path,
        plan=plan,
        plan_ref=plan_ref,
        state_path="artifacts/phase6/state.json",
        manifest_path="artifacts/phase6/manifest.json",
        runner=runner,
        now=lambda: "2026-08-08T00:00:02Z",
    )

    assert manifest["status"] == "completed_with_quarantine"
    assert manifest["jobsCompleted"] == 19
    assert manifest["jobsQuarantined"] == 1
    assert (
        tmp_path / "artifacts/phase6/generation-0001/selfplay/quarantine/pair-0003.json"
    ).is_file()
    assert not (tmp_path / "artifacts/phase6/manifest.json").exists()

    runner.fail_job = None
    resumed = execute_paired_plan(
        repository_root=tmp_path,
        plan=plan,
        plan_ref=plan_ref,
        state_path="artifacts/phase6/state.json",
        manifest_path="artifacts/phase6/manifest.json",
        runner=runner,
        retry_quarantined=True,
        now=lambda: "2026-08-08T00:00:03Z",
    )

    assert resumed["status"] == "completed"
    assert resumed["quarantinedAttempts"] == 1
    assert resumed["jobsCompleted"] == 20
    assert len(runner.calls) == 21


def test_invalid_report_is_quarantined_with_hashed_observed_artifact_evidence(
    tmp_path: Path, selfplay_config: SelfPlayConfig
) -> None:
    plan, plan_ref = _plan(tmp_path, selfplay_config)
    manifest = execute_paired_plan(
        repository_root=tmp_path,
        plan=plan,
        plan_ref=plan_ref,
        state_path="artifacts/phase6/state.json",
        manifest_path="artifacts/phase6/manifest.json",
        runner=FakeRunner(tmp_path, corrupt_job="pair-0003"),
        now=lambda: "2026-08-08T00:00:02Z",
    )

    assert manifest["status"] == "completed_with_quarantine"
    attempt = next(row for row in manifest["attempts"] if row["jobId"] == "pair-0003")
    assert attempt["status"] == "quarantined"
    assert attempt["report"] is None
    assert attempt["csa"] == []
    assert attempt["quarantine"] is not None
    quarantine = load_json(tmp_path / attempt["quarantine"]["path"])
    assert quarantine["observedReport"] is not None
    assert len(quarantine["observedCsa"]) == 2
    assert "CSA identity" in quarantine["failureDetail"]

    observed_csa_path = tmp_path / quarantine["observedCsa"][0]["path"]
    observed_csa_path.write_bytes(b"tampered after quarantine\n")
    with pytest.raises(ContractError, match=r"artifact (?:size|SHA-256) mismatch"):
        execute_paired_plan(
            repository_root=tmp_path,
            plan=plan,
            plan_ref=plan_ref,
            state_path="artifacts/phase6/state.json",
            manifest_path="artifacts/phase6/manifest.json",
            runner=FakeRunner(tmp_path),
            now=lambda: "2026-08-08T00:00:03Z",
        )


def test_interrupted_batch_persists_completed_peer_and_recovers_orphan_logs(
    tmp_path: Path, selfplay_config: SelfPlayConfig
) -> None:
    plan, plan_ref = _plan(tmp_path, selfplay_config)
    orphan_base = tmp_path / "artifacts/phase6/generation-0001/selfplay/logs/pair-0000/attempt-001"
    orphan_base.parent.mkdir(parents=True)
    orphan_base.with_suffix(".stdout.log").write_bytes(b"orphan\n")
    orphan_base.with_suffix(".stderr.log").write_bytes(b"")
    runner = FakeRunner(tmp_path, raise_job="pair-0000")
    arguments = {
        "repository_root": tmp_path,
        "plan": plan,
        "plan_ref": plan_ref,
        "state_path": "artifacts/phase6/state.json",
        "manifest_path": "artifacts/phase6/manifest.json",
        "runner": runner,
        "now": lambda: "2026-08-08T00:00:02Z",
    }

    with pytest.raises(ContractError, match="simulated runner interruption"):
        execute_paired_plan(**arguments)

    state = load_json(tmp_path / "artifacts/phase6/state.json")
    assert state["status"] == "interrupted"
    assert {row["jobId"] for row in state["attempts"]} == {"pair-0000", "pair-0001"}
    assert (
        next(row for row in state["attempts"] if row["jobId"] == "pair-0000")["status"] == "running"
    )
    assert runner.calls.count("pair-0001") == 1
    runner.raise_job = None

    completed = execute_paired_plan(**arguments)

    assert completed["status"] == "completed"
    assert runner.calls.count("pair-0001") == 1
    assert any("recovery-" in row["stdout"]["path"] for row in completed["attempts"])


def test_contained_artifact_paths_reject_parent_traversal(tmp_path: Path) -> None:
    from open_shogi_training.selfplay.common import contained_path

    with pytest.raises(ContractError, match="parent"):
        contained_path(tmp_path, "../outside.json")


def test_execution_rebuilds_planned_starts_from_phase3_source(
    tmp_path: Path, selfplay_config: SelfPlayConfig
) -> None:
    plan, _ = _plan(tmp_path, selfplay_config)
    forged = copy.deepcopy(plan)
    job = forged["jobs"][10]
    job["startPositionId"] = "forged-start"
    seed_material = f"phase6\0{forged['seed']}\0selfplay\0{job['pairIndex']}\0forged-start".encode()
    job["seed"] = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big") % (
        9_007_199_254_740_992
    )
    argv = job["command"]["argv"]
    argv[argv.index("--seed") + 1] = str(job["seed"])
    forged_without_hash = dict(forged)
    forged_without_hash.pop("planSha256")
    forged["planSha256"] = canonical_sha256(forged_without_hash)
    forged_ref = write_ref(
        tmp_path,
        "artifacts/phase6/forged-selfplay-plan.json",
        canonical_json_bytes(forged),
    )

    with pytest.raises(ContractError, match="deterministic train selection"):
        execute_paired_plan(
            repository_root=tmp_path,
            plan=forged,
            plan_ref=forged_ref,
            state_path="artifacts/phase6/forged-state.json",
            manifest_path="artifacts/phase6/forged-manifest.json",
            runner=FakeRunner(tmp_path),
            now=lambda: "2026-08-08T00:00:02Z",
        )


def test_artifact_reference_hash_is_full_file_hash(tmp_path: Path) -> None:
    contents = b"model-bytes"
    reference = write_ref(tmp_path, "weights/model.osaval", contents)

    assert reference.sha256 == hashlib.sha256(contents).hexdigest()


def test_command_runner_allows_only_explicit_repository_commands(tmp_path: Path) -> None:
    executable = tmp_path / "bin" / "open-shogi-cli"
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\nprintf 'validated\\n'\n", encoding="utf-8")
    executable.chmod(0o700)
    runner = CommandRunner(tmp_path)
    executable_ref = ArtifactRef(
        "bin/open-shogi-cli",
        hashlib.sha256(executable.read_bytes()).hexdigest(),
        executable.stat().st_size,
    )

    outcome = runner.run(
        {
            "kind": "engine_validation",
            "argv": ["bin/open-shogi-cli", "perft", "--depth", "0", "--sfen", SFEN],
            "timeoutSeconds": 5,
        },
        stdout_path="artifacts/validation.stdout",
        stderr_path="artifacts/validation.stderr",
        expected_executable=executable_ref,
    )

    assert outcome.return_code == 0
    assert (tmp_path / outcome.stdout.path).read_text(encoding="utf-8") == "validated\n"
    with pytest.raises(ContractError, match="exact repository model module"):
        runner.run(
            {
                "kind": "model_cli",
                "argv": ["python", "-m", "untrusted.module", "train"],
                "timeoutSeconds": 5,
            },
            stdout_path="artifacts/rejected.stdout",
            stderr_path="artifacts/rejected.stderr",
        )
    with pytest.raises(ContractError, match="unknown or duplicate option"):
        runner.run(
            {
                "kind": "engine_validation",
                "argv": [
                    "bin/open-shogi-cli",
                    "perft",
                    "--depth",
                    "0",
                    "--sfen",
                    SFEN,
                    "--unplanned",
                    "value",
                ],
                "timeoutSeconds": 5,
            },
            stdout_path="artifacts/unknown.stdout",
            stderr_path="artifacts/unknown.stderr",
        )
    with pytest.raises(ContractError, match="command input must be"):
        runner.run(
            {
                "kind": "model_cli",
                "argv": [
                    "python",
                    "-m",
                    "open_shogi_training.models",
                    "train",
                    "--features",
                    "configs/missing-features.toml",
                    "--model",
                    "configs/missing-model.toml",
                    "--training",
                    "configs/missing-training.toml",
                    "--labels",
                    "artifacts/missing-labels.jsonl",
                    "--positions",
                    "data/missing-positions.jsonl.gz",
                    "--dataset-manifest",
                    "data/missing-manifest.json",
                    "--replay-manifest",
                    "artifacts/missing-replay.json",
                    "--output-dir",
                    "artifacts/training",
                ],
                "timeoutSeconds": 5,
            },
            stdout_path="artifacts/missing.stdout",
            stderr_path="artifacts/missing.stderr",
        )


def test_command_runner_cleanup_failure_closes_all_resources_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "bin/open-shogi-cli"
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\nprintf 'validated\\n'\n", encoding="utf-8")
    executable.chmod(0o700)
    executable_ref = artifact_ref(tmp_path, "bin/open-shogi-cli")
    close_calls: list[str] = []

    class FailingSnapshot:
        executable_path = str(executable)

        @staticmethod
        def pass_fds() -> tuple[int, ...]:
            return ()

        @staticmethod
        def assert_snapshot_unchanged() -> None:
            pass

        @staticmethod
        def assert_source_unchanged() -> None:
            pass

        @staticmethod
        def close() -> None:
            close_calls.append("snapshot")
            raise OSError("forced snapshot cleanup failure")

    monkeypatch.setattr(
        execution_module.ExecutableSnapshot,
        "create",
        lambda *_args, **_kwargs: FailingSnapshot(),
    )
    stdout = "artifacts/unpublished.stdout"
    stderr = "artifacts/unpublished.stderr"
    receipt = "artifacts/unpublished.receipt.json"

    with pytest.raises(OSError, match="forced snapshot cleanup failure"):
        CommandRunner(tmp_path).run(
            {
                "kind": "engine_validation",
                "argv": ["bin/open-shogi-cli", "perft", "--depth", "0", "--sfen", SFEN],
                "timeoutSeconds": 5,
            },
            stdout_path=stdout,
            stderr_path=stderr,
            receipt_path=receipt,
            expected_executable=executable_ref,
        )

    assert close_calls == ["snapshot"]
    assert not (tmp_path / stdout).exists()
    assert not (tmp_path / stderr).exists()
    assert not (tmp_path / receipt).exists()


def test_bounded_process_stops_immediately_when_rss_enforcement_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_measurement(_pid: int):
        raise ContractError("measurement unavailable")

    monkeypatch.setattr(execution_module, "_process_tree_rss_bytes", fail_measurement)
    started = time.monotonic()
    outcome = execution_module._run_bounded_process(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        timeout_seconds=10,
        memory_limit_bytes=128 * 1024 * 1024,
    )

    assert time.monotonic() - started < 5
    assert outcome[5] is True
    assert outcome[7] == "unavailable"


def test_bounded_process_checks_the_launch_snapshot_before_and_after_spawn(
    tmp_path: Path,
) -> None:
    checks: list[int] = []

    outcome = execution_module._run_bounded_process(
        ["/usr/bin/true"],
        cwd=tmp_path,
        timeout_seconds=5,
        memory_limit_bytes=128 * 1024 * 1024,
        launch_guard=lambda: checks.append(len(checks) + 1),
    )

    assert outcome[2] == 0
    assert checks == [1, 2]


@pytest.mark.parametrize("process_identity_available", [True, False])
def test_bounded_process_terminates_child_when_post_spawn_guard_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    process_identity_available: bool,
) -> None:
    checks = 0

    if not process_identity_available:
        monkeypatch.setattr(execution_module, "_process_start_identity", lambda _pid: None)

    def guard() -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise ContractError("private launch snapshot changed")

    started = time.monotonic()
    with pytest.raises(ContractError, match="private launch snapshot changed"):
        execution_module._run_bounded_process(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=tmp_path,
            timeout_seconds=30,
            memory_limit_bytes=128 * 1024 * 1024,
            launch_guard=guard,
        )

    assert checks == 2
    assert time.monotonic() - started < 5


def test_bounded_process_retries_transient_leader_identity_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_identity = execution_module._process_start_identity
    attempts = 0

    def transient_identity(process_id: int) -> str | None:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            return None
        return real_identity(process_id)

    monkeypatch.setattr(execution_module, "_process_start_identity", transient_identity)
    outcome = execution_module._run_bounded_process(
        [sys.executable, "-c", "import time; time.sleep(.2)"],
        cwd=tmp_path,
        timeout_seconds=5,
        memory_limit_bytes=128 * 1024 * 1024,
    )

    assert outcome[2] == 0
    assert outcome[5] is False
    assert attempts >= 3


def test_bounded_process_fails_closed_when_leader_identity_never_binds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(execution_module, "_process_start_identity", lambda _pid: None)
    started = time.monotonic()

    with pytest.raises(
        ContractError, match="cannot bind repository command process-start identity"
    ):
        execution_module._run_bounded_process(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=tmp_path,
            timeout_seconds=30,
            memory_limit_bytes=128 * 1024 * 1024,
        )

    assert time.monotonic() - started < 5


def test_bounded_process_allows_only_an_explicit_short_lived_no_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def sample_after_true_has_had_time_to_exit(_pid: int) -> tuple[None, set[int]]:
        # Model the narrow real-world race in which `ps` returns after the
        # process has exited without ever yielding a measurable row.
        time.sleep(0.1)
        return None, set()

    monkeypatch.setattr(
        execution_module, "_process_tree_rss_bytes", sample_after_true_has_had_time_to_exit
    )

    outcome = execution_module._run_bounded_process(
        ["/usr/bin/true"],
        cwd=tmp_path,
        timeout_seconds=5,
        memory_limit_bytes=128 * 1024 * 1024,
    )

    assert outcome[2] == 0
    assert outcome[5] is False
    assert outcome[6] == 0
    assert outcome[7] == "process_tree_ps_short_lived_no_sample"


def test_bounded_process_rejects_no_sample_while_leader_is_still_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(execution_module, "_process_tree_rss_bytes", lambda _pid: (None, set()))

    outcome = execution_module._run_bounded_process(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        timeout_seconds=10,
        memory_limit_bytes=128 * 1024 * 1024,
    )

    assert outcome[5] is True
    assert outcome[7] == "unavailable"


def test_process_tree_signals_refuse_reused_process_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LiveProcess:
        pid = 43_210

        @staticmethod
        def poll() -> None:
            return None

    group_signals: list[tuple[int, int]] = []
    process_signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        execution_module,
        "_process_start_identity",
        lambda _process_id: "new",
    )
    monkeypatch.setattr(
        execution_module.os,
        "killpg",
        lambda process_group, signal_number: group_signals.append((process_group, signal_number)),
    )
    monkeypatch.setattr(
        execution_module.os,
        "kill",
        lambda process_id, signal_number: process_signals.append((process_id, signal_number)),
    )

    assert not execution_module._signal_owned_process_group(
        LiveProcess(),  # type: ignore[arg-type]
        signal.SIGTERM,
        {43_210: "old"},
    )
    execution_module._signal_process_ids(
        {43_211},
        signal.SIGKILL,
        exclude=set(),
        identities={43_211: "old"},
    )

    assert group_signals == []
    assert process_signals == []


def test_rss_snapshot_counts_an_exited_descendant_without_retaining_its_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leader = 43_210
    exited = 43_211
    monkeypatch.setattr(
        execution_module.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["/bin/ps"],
            returncode=0,
            stdout=(f"{leader} 1 {leader} 100\n{exited} {leader} {leader} 50\n").encode(),
            stderr=b"",
        ),
    )
    monkeypatch.setattr(
        execution_module,
        "_process_start_identity",
        lambda process_id: "leader-identity" if process_id == leader else None,
    )

    def process_exists(process_id: int, signal_number: int) -> None:
        assert signal_number == 0
        if process_id == exited:
            raise ProcessLookupError

    monkeypatch.setattr(execution_module.os, "kill", process_exists)

    rss, live = execution_module._process_tree_rss_bytes(leader)

    assert rss == 150 * 1024
    assert live == {leader}
    assert live.identities == {leader: "leader-identity"}


def test_rss_snapshot_counts_a_zombie_descendant_without_retaining_its_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leader = 43_210
    zombie = 43_211
    snapshots = iter(
        (
            subprocess.CompletedProcess(
                args=["/bin/ps"],
                returncode=0,
                stdout=(f"{leader} 1 {leader} 100\n{zombie} {leader} {leader} 50\n").encode(),
                stderr=b"",
            ),
            subprocess.CompletedProcess(
                args=["/bin/ps"],
                returncode=0,
                stdout=f"{zombie} {leader} Z\n".encode(),
                stderr=b"",
            ),
        )
    )
    monkeypatch.setattr(
        execution_module.subprocess,
        "run",
        lambda *_args, **_kwargs: next(snapshots),
    )
    monkeypatch.setattr(
        execution_module,
        "_process_start_identity",
        lambda process_id: "leader-identity" if process_id == leader else None,
    )
    monkeypatch.setattr(execution_module.os, "kill", lambda _pid, _signal: None)

    rss, live = execution_module._process_tree_rss_bytes(leader)

    assert rss == 150 * 1024
    assert live == {leader}
    assert live.identities == {leader: "leader-identity"}


def test_bounded_process_terminates_a_descendant_that_escapes_the_process_group(
    tmp_path: Path,
) -> None:
    program = (
        "import subprocess,sys,time;"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],"
        "start_new_session=True);"
        "print(child.pid,flush=True);time.sleep(30)"
    )
    outcome = execution_module._run_bounded_process(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        timeout_seconds=1,
        memory_limit_bytes=512 * 1024 * 1024,
    )

    child_pid = int(outcome[0].decode().strip())
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("escaped descendant remained alive after bounded-process termination")
    assert outcome[3] is True


def test_bounded_process_kills_a_sampled_descendant_after_leader_exit_and_reparent(
    tmp_path: Path,
) -> None:
    program = (
        "import subprocess,sys,time;"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],"
        "start_new_session=True);"
        "print(child.pid,flush=True);time.sleep(0.5)"
    )
    outcome = execution_module._run_bounded_process(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        timeout_seconds=5,
        memory_limit_bytes=512 * 1024 * 1024,
    )

    child_pid = int(outcome[0].decode().strip())
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("reparented descendant remained alive after its leader exited")
    assert outcome[2] == 0
    assert outcome[3] is False


@pytest.mark.parametrize("coordinator_shutdown", [False, True])
def test_pipe_reader_classifies_only_unplanned_close_as_invalid_output(
    coordinator_shutdown: bool,
) -> None:
    class ClosedStream:
        def read(self, _size: int) -> bytes:
            raise ValueError("closed")

        def close(self) -> None:
            raise ValueError("already closed")

    overflow = threading.Event()
    shutdown = threading.Event()
    if coordinator_shutdown:
        shutdown.set()

    execution_module._read_pipe(ClosedStream(), bytearray(), overflow, shutdown)

    assert overflow.is_set() is not coordinator_shutdown
