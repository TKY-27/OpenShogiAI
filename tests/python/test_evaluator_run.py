"""Sealing, real lease inheritance, owned-process cleanup and checkpoint-safe supervision."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from open_shogi_training import evaluator_run as runner
from open_shogi_training.evaluator_data import atomic, digest, encoded

PROJECT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "scalar,screen,arena",
    [(False, True, True), (True, False, True), (True, True, False), (True, True, True)],
)
def test_candidate_review_requires_all_independent_gates(
    tmp_path, monkeypatch, scalar, screen, arena
):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    atomic(tmp_path / "run.json", encoded({"fixture": True}))
    atomic(tmp_path / "fit/best.osaval03", b"fixture")
    atomic(tmp_path / "arena/arena.json", encoded({"adoption_criteria_met": arena}))
    atomic(
        tmp_path / "development-test.json",
        encoded(
            {
                "groups": {
                    "r3": {g: {"loss": 1.0} for g in ("general", "attack_end")},
                    "candidate": {
                        g: {"loss": 1.02 if scalar else 1.04} for g in ("general", "attack_end")
                    },
                },
                "move_quality_screen": {"screen_pass": screen},
            }
        ),
    )
    review = runner._candidate_review(tmp_path)
    assert review["meets_frozen_criteria"] is (scalar and screen and arena)
    assert review["promotion_performed"] is False
    assert review["human_shodan_validated"] is False


def put(path: Path, content: bytes = b"fixture") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    config = json.loads((PROJECT / "configs/evaluator-main.json").read_text())
    config["generation"].pop("defense_campaign", None)
    config["generation"].pop("recovery_policy", None)
    config.pop("resource_epoch", None)
    config["resources"].pop("minimum_memory_free_percent", None)
    config["evaluation"].pop("groups", None)
    config["evaluation"]["startpos_demonstration_games"] = 8
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    config.update(run_id="contract-test", output="local/runs/contract-test")
    generation = config["generation"]
    generation.update(games=1, first_game=22, workers=1)
    generation["leaf_sha256"] = digest(put(tmp_path / generation["leaf_path"]))
    put(tmp_path / generation["teacher_config_path"], b"teacher: fixture\n")
    put(tmp_path / generation["split_guard_path"], b"{}")
    generation["development_exclusions_path"] = "local/diagnosis/exclusions.json"
    generation["development_exclusions_sha256"] = digest(
        put(tmp_path / generation["development_exclusions_path"], b'{"symmetry_keys":[]}')
    )
    for name, path in runner.RUNTIME_SOURCES.items():
        put(tmp_path / path, name.encode())
    put(tmp_path / "scripts/check_evaluator_model.mjs", b"// audit fixture\n")
    tracked = put(tmp_path / "training/runner-fixture.py", b"# sealed code\n")
    code = {"commit": "a" * 40, "files": {str(tracked.relative_to(tmp_path)): digest(tracked)}}
    monkeypatch.setattr(runner, "_code_identity", lambda: code)
    source = put(tmp_path / "configs/evaluator-main.json", encoded(config))
    result = runner.seal(source)
    run = tmp_path / config["output"]
    return tmp_path, run, runner.verify(run), source, result


def test_seal_pins_config_runtime_inputs_and_code(prepared):
    root, run, config, _, receipt = prepared
    assert receipt["run_sha256"] == digest(run / "run.json")
    assert config["generation"]["replay_path"].startswith("local/runs/contract-test/runtime/")
    assert runner.verify(run)["run_id"] == "contract-test"
    put(root / config["generation"]["development_exclusions_path"], b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        runner.verify(run)


@pytest.mark.parametrize("field", ["training", "resources", "generation"])
def test_sealed_run_config_mutation_is_rejected(prepared, field):
    _, run, config, _, _ = prepared
    config[field]["unexpected_change"] = True
    atomic(run / "run.json", encoded(config))
    with pytest.raises(ValueError, match=r"sealed run\.json changed"):
        runner.verify(run)


def test_runtime_and_commit_file_drift_are_detected(prepared):
    root, run, config, _, _ = prepared
    runtime = root / config["runtime"]["wasm"]["path"]
    previous = runtime.read_bytes()
    runtime.write_bytes(b"new wasm")
    with pytest.raises(ValueError, match="artifact changed"):
        runner.verify(run)
    runtime.write_bytes(previous)
    put(root / next(iter(config["code"]["files"])), b"edited implementation")
    with pytest.raises(ValueError, match="artifact changed"):
        runner.verify(run)


def test_partial_seal_never_publishes_a_run_or_blocks_retry(prepared, monkeypatch):
    root, _, config, source, _ = prepared
    original = json.loads(source.read_text())
    original.update(run_id="second-run", output="local/runs/second-run")
    source = put(root / "configs/second.json", encoded(original))
    real_copy = runner.shutil.copy2

    def failed_copy(*args, **kwargs):
        real_copy(*args, **kwargs)
        raise OSError("simulated interrupted copy")

    monkeypatch.setattr(runner.shutil, "copy2", failed_copy)
    with pytest.raises(OSError, match="interrupted copy"):
        runner.seal(source)
    assert not (root / original["output"]).exists()
    assert not list((root / "local/runs").glob(".second-run.sealing-*"))
    monkeypatch.setattr(runner.shutil, "copy2", real_copy)
    assert runner.seal(source)["status"] == "prepared"
    assert runner.verify(root / original["output"])["code"] == config["code"]


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("resources", "maximum_wall_seconds", float("inf")),
        ("resources", "monitor_interval_seconds", 0),
        ("resources", "maximum_retries", 100000),
        ("generation", "workers", True),
        ("generation", "games", 0),
        ("training", "learning_rate", float("nan")),
        ("training", "minimum_learning_rate", 10),
        ("training", "max_epochs", 0),
    ],
)
def test_invalid_or_unbounded_execution_settings_are_rejected(prepared, section, key, value):
    _, _, config, _, _ = prepared
    changed = copy.deepcopy(config)
    changed[section][key] = value
    with pytest.raises(ValueError):
        runner._validate_config(changed)


def test_rejects_uncommitted_new_run_code(monkeypatch):
    monkeypatch.setattr(
        runner.subprocess, "check_output", lambda *_args, **_kwargs: "?? training/new.py\n"
    )
    with pytest.raises(ValueError, match="including new files"):
        runner._code_identity()


def dataset(run: Path, config: dict, root: Path) -> Path:
    data, folder = run / "data", run / "data/dataset"
    atomic(data / "generation.json", encoded(config["generation"]))
    atomic(data / "generation-complete.json", encoded({"games": 1}))
    game = put(data / "games/000022.json.gz", b"source trajectory bytes")
    array = put(folder / "train-targets.npy", b"encoded array bytes")
    manifest = {
        "split_guard_sha256": config["generation"]["split_guard_sha256"],
        "generation_sha256": digest(data / "generation.json"),
        "exclusions_sha256": hashlib.sha256(
            encoded(sorted(config["excluded_development_sfens"]))
        ).hexdigest(),
        "source_games": [{"path": game.name, "sha256": digest(game)}],
        "artifacts": [{"path": array.name, "sha256": digest(array)}],
        "unique_positions": {"train": 1, "validation": 1, "development_test": 1},
    }
    atomic(folder / "manifest.json", encoded(manifest))
    assert root == runner.ROOT
    return array


def completion(run: Path, stage: str, result: dict):
    atomic(
        run / f"{stage}-complete.json",
        encoded(
            {
                "schema": "open_shogiai_evaluator_stage/v1",
                "stage": stage,
                "run_sha256": digest(run / "run.json"),
                "result": result,
                "artifacts": runner._completion_artifacts(run, stage),
            }
        ),
    )


def test_resume_validates_array_bytes_not_only_the_manifest(prepared):
    root, run, config, _, _ = prepared
    array = dataset(run, config, root)
    result = runner._dataset(run, config)
    completion(run, "prepare", result)
    assert runner._verify_completion(run, "prepare", config) == result
    array.write_bytes(b"changed labels while keeping manifest")
    with pytest.raises(ValueError, match="dataset corrupted"):
        runner._verify_completion(run, "prepare", config)


def test_completion_receipt_cannot_skip_missing_or_changed_artifacts(prepared):
    _, run, config, _, _ = prepared
    put(run / "data/generation.json", encoded(config["generation"]))
    put(run / "data/generation-complete.json", encoded({"games": 1}))
    archive = put(run / "data/games/000022.json.gz", b"completed game")
    completion(run, "generate", {"games": 1})
    archive.unlink()
    with pytest.raises(ValueError, match="artifact changed"):
        runner._verify_completion(run, "generate", config)


def audit_report(root: Path, run: Path, config: dict) -> Path:
    model = put(run / "fit/best.osaval03", b"best model")
    paths = {
        "auditScript": root / "scripts/check_evaluator_model.mjs",
        "model": model,
        "moduleJs": root / config["runtime"]["module"]["path"],
        "wasm": root / config["runtime"]["wasm"]["path"],
        "nativeProbe": root / config["runtime"]["replay"]["path"],
    }
    report = {
        "schema": "open_shogiai_evaluator_export_audit/v1",
        "runtime": "actual-wasm-in-node",
        "searches": [{}, {}],
        "status": "PASS",
        "format": "OSAVAL03",
        "profile": "pure_learned",
        "errors": [],
        "parity": {"errors": 0, "roots": 10, "children": 300, "maximumCpDifference": 0},
        "artifacts": {name: {"sha256": digest(path)} for name, path in paths.items()},
    }
    return put(run / "model-audit.json", encoded(report))


def test_interrupted_audit_can_reuse_only_a_matching_success_report(prepared):
    root, run, config, _, _ = prepared
    report = audit_report(root, run, config)
    identity = digest(report)
    assert runner._audit_report(run, config)["status"] == "PASS"
    assert digest(report) == identity
    put(run / "fit/best.osaval03", b"different candidate")
    with pytest.raises(ValueError, match="artifact changed"):
        runner._audit_report(run, config)


@pytest.mark.parametrize(
    ("status", "all_complete", "error"),
    [
        ("stopped", False, InterruptedError),
        ("failed", False, ValueError),
        ("complete", False, ValueError),
    ],
)
def test_unfinished_arena_never_gets_a_completion_receipt(status, all_complete, error):
    with pytest.raises(error):
        runner._arena_complete(
            {
                "status": status,
                "planned_games": 40,
                "summary": {"all_planned_complete": all_complete},
            }
        )


def test_valid_losing_arena_does_not_require_a_new_strength_gate():
    runner._arena_complete(
        {
            "status": "complete",
            "planned_games": 40,
            "adoption_criteria_met": False,
            "summary": {"all_planned_complete": True},
        }
    )


def test_real_inherited_lease_survives_supervisor_descriptor_closure(prepared):
    _, run, _, _, _ = prepared
    child = None
    try:
        with runner._lease(run) as descriptor:
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                pass_fds=(descriptor,),
                start_new_session=True,
            )
        # Equivalent lease lifetime to an abruptly lost supervisor: only the child holds it.
        with pytest.raises(BlockingIOError), runner._lease(run):
            pytest.fail("a second writer acquired the live stage lease")
        child.terminate()
        child.wait(timeout=3)
        with runner._lease(run):
            pass
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=3)


def test_stale_pid_identity_is_never_signalled(monkeypatch):
    monkeypatch.setattr(runner, "read_process_identity", lambda _pid: "new-process")
    sent = []
    monkeypatch.setattr(runner.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    assert runner._signal_owned({999999: "old-process"}, signal.SIGKILL) == []
    assert sent == []


def test_real_cleanup_finds_an_unobserved_orphan_before_reaping_its_stage(prepared):
    _, _, _, _, _ = prepared
    script = (
        "import subprocess,sys,os; "
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); "
        "print(p.pid,flush=True); os._exit(17)"
    )
    leader = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True, start_new_session=True
    )
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(60)"], start_new_session=True
    )
    child_pid = None
    try:
        assert select.select([leader.stdout], [], [], 3)[0], "child startup timed out"
        child_pid = int(leader.stdout.readline())
        deadline = time.monotonic() + 3
        while not runner._process_table()[leader.pid][2].startswith("Z"):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        # No ownership sample occurred while either creator was alive.
        assert leader.returncode is None
        assert runner._process_table()[child_pid][0] != leader.pid
        result = runner._cleanup(leader, {}, grace_seconds=0)
        assert leader.returncode == 17
        assert result["group_cleanup_before_reap"] and not result["remaining_processes"]
        assert child_pid in result["signalled_pids"]
        assert unrelated.poll() is None
    finally:
        # These PIDs/groups remain reserved by our own unreaped direct children.
        if leader.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(leader.pid, signal.SIGKILL)
        for process in (leader, unrelated):
            if process.poll() is None:
                process.kill()
            process.wait(timeout=3)
        leader.stdout.close()


def test_group_cleanup_refuses_a_reaped_leader_before_any_signal(monkeypatch):
    process = FakeProcess()
    process.returncode = 0
    monkeypatch.setattr(runner.os, "killpg", lambda *_args: pytest.fail("unsafe group signal"))
    with pytest.raises(RuntimeError, match="reaped before"):
        runner._cleanup(process, {}, grace_seconds=0)


def test_group_cleanup_refuses_unproved_or_reused_group_identity(monkeypatch):
    process = FakeProcess()
    monkeypatch.setattr(
        runner, "_process_table", lambda: {process.pid: (os.getpid(), 100, "S", process.pid)}
    )
    monkeypatch.setattr(runner, "read_process_identity", lambda _pid: "changed-identity")
    monkeypatch.setattr(runner.os, "killpg", lambda *_args: pytest.fail("unsafe group signal"))
    with pytest.raises(RuntimeError, match="identity changed"):
        runner._cleanup(process, {process.pid: "original-identity"}, grace_seconds=0)


def test_start_rejects_residual_group_even_after_its_leader_has_disappeared(prepared, monkeypatch):
    _, run, _, _, _ = prepared
    atomic(
        run / "stage-process.json",
        encoded(
            {
                "schema": "open_shogiai_evaluator_stage_process/v1",
                "run_sha256": digest(run / "run.json"),
                "stage": "generate",
                "pid": 888888,
                "pgid": 888888,
                "process_identity": "original",
                "status": "active",
            }
        ),
    )
    # The stage lease was lost and the orphan no longer has the original PPID.
    monkeypatch.setattr(runner, "_process_table", lambda: {999999: (1, 100, "S", 888888)})
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_a, **_k: pytest.fail("second writer"))
    monkeypatch.setattr(runner.os, "killpg", lambda *_a: pytest.fail("unproved ownership signal"))
    result = runner.start(run)
    assert result["status"] == "needs_astra"
    assert result["reason"] == "residual_stage_group_unproven"
    assert result["residual_group"]["remaining_pids"] == [999999]
    assert runner.work(run)["reason"] == "residual_stage_group_unproven"


def test_real_lost_supervisor_and_stage_cannot_hide_an_orphan_from_restart(prepared):
    root, run, _, _, _ = prepared
    stage_code = "\n".join(
        [
            "import os,subprocess,sys,time",
            "from pathlib import Path",
            f"sys.path.insert(0, {str(PROJECT / 'training')!r})",
            "from open_shogi_training import evaluator_run as r",
            f"r.ROOT=Path({str(root)!r}); run=Path({str(run)!r})",
            "with r._lease(run,int(sys.argv[1])):",
            " r._register_stage_group(run,'generate')",
            " child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])",
            " r.atomic(run/'child.pid',str(child.pid).encode())",
            " while not (run/'allow-stage-exit').exists(): time.sleep(.01)",
            " os._exit(19)",
        ]
    )
    supervisor_code = "\n".join(
        [
            "import subprocess,sys,time",
            "from pathlib import Path",
            f"sys.path.insert(0, {str(PROJECT / 'training')!r})",
            "from open_shogi_training import evaluator_run as r",
            f"r.ROOT=Path({str(root)!r}); run=Path({str(run)!r})",
            "with r._lease(run) as lease:",
            f" p=subprocess.Popen([sys.executable,'-c',{stage_code!r},str(lease)],"
            "start_new_session=True,pass_fds=(lease,))",
            " time.sleep(60)",
        ]
    )
    supervisor = subprocess.Popen([sys.executable, "-c", supervisor_code], start_new_session=True)
    owned = {}
    try:
        deadline = time.monotonic() + 5
        while not (run / "child.pid").exists():
            assert time.monotonic() < deadline, "registered stage startup timed out"
            time.sleep(0.02)
        child_pid = int((run / "child.pid").read_text())
        receipt = runner._json(run / "stage-process.json")
        for pid in (child_pid, receipt["pid"]):
            identity = runner.read_process_identity(pid)
            assert identity is not None
            owned[pid] = identity
        supervisor.kill()
        supervisor.wait(timeout=3)
        (run / "allow-stage-exit").touch()
        deadline = time.monotonic() + 3
        while runner.read_process_identity(receipt["pid"]) is not None:
            assert time.monotonic() < deadline
            time.sleep(0.02)
        with runner._lease(run):
            pass  # Neither supervisor nor stage remains to hold the lease.
        result = runner.start(run)
        assert result["reason"] == "residual_stage_group_unproven"
        assert child_pid in result["residual_group"]["remaining_pids"]
        assert runner.read_process_identity(child_pid) == owned[child_pid]
    finally:
        runner._signal_owned(owned, signal.SIGKILL)
        if supervisor.poll() is None:
            supervisor.kill()
        supervisor.wait(timeout=3)


def test_stage_records_its_group_before_starting_generation(prepared, monkeypatch):
    _, run, config, _, _ = prepared
    monkeypatch.setattr(runner.os, "getpgid", lambda _pid: os.getpid())
    monkeypatch.setattr(runner.os, "getsid", lambda _pid: os.getpid())
    monkeypatch.setattr(runner, "read_process_identity", lambda _pid: "stage-birth-identity")

    def generation(*_args):
        receipt = runner._json(run / "stage-process.json")
        assert receipt["pid"] == receipt["pgid"] == os.getpid()
        assert receipt["run_sha256"] == digest(run / "run.json")
        assert receipt["process_identity"] == "stage-birth-identity"
        raise InterruptedError("before child launch")

    monkeypatch.setattr(runner, "generate", generation)
    with pytest.raises(InterruptedError, match="before child launch"):
        runner.stage_run(run, "generate")
    assert config["generation"]["games"] == 1


def test_arena_completion_retains_stderr_when_failed_attempt_has_no_trace(prepared):
    root, run, _, _, _ = prepared
    stderr = put(run / "arena/attempt-00.stderr.log", b"failed before first move")
    put(run / "arena/plan.json", b"{}")
    put(
        run / "arena/arena.json",
        encoded({"attempts": [{"trace": None, "stderr_artifacts": [runner._reference(stderr)]}]}),
    )
    refs = runner._completion_artifacts(run, "arena")
    assert refs[-1]["path"] == str(stderr.relative_to(root))
    assert len(refs) == 3


def test_atomic_progress_publication_can_disappear_between_listing_and_stat(prepared, monkeypatch):
    _, run, _, _, _ = prepared
    path = put(run / "fit/checkpoint-000001.pt", b"old checkpoint")
    real_lstat = Path.lstat

    def vanished(self, *args, **kwargs):
        if self == path:
            raise FileNotFoundError("concurrent checkpoint pruning")
        return real_lstat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", vanished)
    assert runner._progress_signature(run, "train") == (0, 0, 0)


class FakeProcess:
    pid = 900001
    returncode = None

    def poll(self):
        return self.returncode


def stage_state():
    return {"retries": {}, "began_at": time.time(), "initial_swap_bytes": 0}


def test_monitor_exception_stops_and_cleans_the_owned_stage(prepared, monkeypatch):
    _, run, config, _, _ = prepared
    process, cleaned = FakeProcess(), []
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_a, **_kw: process)
    monkeypatch.setattr(runner, "read_process_identity", lambda _pid: "test-identity")
    monkeypatch.setattr(
        runner,
        "_stage_group_snapshot",
        lambda *_a: {process.pid: (os.getpid(), 10, "S", process.pid)},
    )

    def failed_progress(*_args):
        raise OSError("simulated monitor filesystem failure")

    monkeypatch.setattr(runner, "_progress_signature", failed_progress)

    def cleanup(observed, owned, **_kwargs):
        cleaned.append((observed, owned))
        process.returncode = -signal.SIGTERM
        return {"remaining_processes": {}, "signalled_pids": [process.pid], "inventory_errors": []}

    monkeypatch.setattr(runner, "_cleanup", cleanup)
    state = stage_state()
    with runner._lease(run) as lease:
        code, failure = runner._run_stage(run, "train", config, state, lease)
    assert code == -signal.SIGTERM and failure.startswith("supervision_error:")
    assert cleaned and (run / "STOP").exists()
    assert state["cleanup"]["remaining_processes"] == {}


@pytest.mark.parametrize("returncode,expected", [(1, None), (runner.PAUSED_EXIT, "requested_stop")])
def test_stop_never_hides_a_child_failure_before_monitor_tick(
    prepared, monkeypatch, returncode, expected
):
    _, run, config, _, _ = prepared
    process = FakeProcess()
    process.returncode = returncode

    def launch(*_args, **_kwargs):
        runner.stop(run)
        return process

    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    monkeypatch.setattr(runner, "read_process_identity", lambda _pid: "test-identity")
    monkeypatch.setattr(
        runner,
        "_stage_group_snapshot",
        lambda *_a: {process.pid: (os.getpid(), 0, "Z", process.pid)},
    )
    monkeypatch.setattr(runner, "_cleanup", lambda *_a, **_kw: {"remaining_processes": {}})
    with runner._lease(run) as lease:
        code, failure = runner._run_stage(run, "train", config, stage_state(), lease)
    assert (code, failure) == (returncode, expected)


def test_retry_is_finite_and_does_not_restart_contract_failures(prepared, monkeypatch):
    _, run, _, _, _ = prepared
    monkeypatch.setattr(runner, "_swap_bytes", lambda: 0)
    launches = []

    def interrupted(*_args):
        launches.append(1)
        return -signal.SIGTERM, None

    monkeypatch.setattr(runner, "_run_stage", interrupted)
    result = runner.work(run)
    assert result["status"] == "needs_astra"
    assert len(launches) == 2
    assert result["retries"] == {"generate": 1}
    # A run requiring Astra cannot be relaunched by either execution entry point.
    runner.work(run)
    assert len(launches) == 2
    assert runner._json(run / "state.json")["retries"] == {"generate": 1}
    monkeypatch.setattr(runner, "_run_stage", lambda *_args: (1, "data_corruption"))
    assert runner.work(run)["reason"] == "stage_exit_-15"


def test_foreign_run_state_and_review_stage_are_rejected(prepared):
    _, run, _, _, _ = prepared
    state = runner._json(run / "state.json")
    atomic(run / "state.json", encoded({**state, "run_id": "other-run"}))
    with pytest.raises(ValueError, match="another contract"):
        runner._state(run)
    atomic(run / "state.json", encoded({**state, "status": "needs_astra"}))
    with pytest.raises(ValueError, match="Astra review"):
        runner.stage_run(run, "generate")


def test_defense_contract_rejects_legacy_state_and_terminal_transitions(prepared):
    _, run, _, _, _ = prepared
    config = runner._json(run / "run.json")
    reviewed = json.loads((PROJECT / "configs/evaluator-main.json").read_text())
    reviewed["state_machine"]["transitions"]["awaiting_astra_review"] = ["running"]
    with pytest.raises(ValueError, match="transitions"):
        runner._validate_config(reviewed)
    config["generation"]["defense_campaign"] = {"fixture": True}
    config["state_machine"] = reviewed["state_machine"]
    atomic(run / "run.json", encoded(config))
    with pytest.raises(ValueError, match="outside this defense"):
        runner._state(run)


def test_review_status_detects_changed_candidate_evidence(prepared):
    _, run, _, _, _ = prepared
    paths = {
        "run_sha256": run / "run.json",
        "candidate_sha256": put(run / "fit/best.osaval03"),
        "offline_sha256": put(run / "development-test.json"),
        "arena_sha256": put(run / "arena/arena.json"),
    }
    atomic(run / "candidate-review.json", encoded({k: digest(p) for k, p in paths.items()}))
    state = runner._state(run)
    state.update(
        status="awaiting_astra_review", review_sha256=digest(run / "candidate-review.json")
    )
    atomic(run / "state.json", encoded(state))
    assert runner.status(run)["status"] == "awaiting_astra_review"
    paths["candidate_sha256"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        runner.status(run)


def test_audit_callback_resumes_after_report_before_completion_without_relaunching_node(
    prepared, monkeypatch
):
    import torch
    from open_shogi_training import evaluator_training, phase10v_model

    root, run, config, _, _ = prepared
    report = audit_report(root, run, config)
    previous = digest(report)
    monkeypatch.setattr(runner, "_register_stage_group", lambda *_args: {})
    monkeypatch.setattr(runner, "_dataset", lambda *_args: {})
    monkeypatch.setattr(runner, "_training_summary", lambda *_args: {})
    monkeypatch.setattr(evaluator_training, "arrays", lambda _folder, split: {"split": split})
    monkeypatch.setattr(evaluator_training, "evaluate", lambda *_args: {"loss": 1.0})
    monkeypatch.setattr(phase10v_model.Phase10VModel, "read", lambda path: path)
    monkeypatch.setattr(phase10v_model, "torch_parameters", lambda model: model)
    monkeypatch.setattr(torch, "set_num_threads", lambda _threads: None)

    def unexpected_process(*_args, **_kwargs):
        pytest.fail("successful audit was launched again instead of checking its identities")

    monkeypatch.setattr(runner.subprocess, "run", unexpected_process)
    result = runner.stage_run(run, "audit")
    assert result["audit_sha256"] == previous
    assert (run / "audit-complete.json").exists()
    assert digest(report) == previous
    assert runner.stage_run(run, "audit") == result


def test_dataset_reference_cannot_escape_before_prepare_opens_it(prepared):
    root, run, config, _, _ = prepared
    dataset(run, config, root)
    path = run / "data/dataset/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["artifacts"][0]["path"] = "../../../../sealed-evaluation.json"
    atomic(path, encoded(manifest))
    with pytest.raises(ValueError, match="local file names"):
        runner._dataset(run, config)


def test_probe_preserves_resource_baselines_and_does_not_enter_training(prepared, monkeypatch):
    _, run, _, _, _ = prepared
    prior = runner._state(run)
    prior.update(began_at=time.time() - 600, initial_swap_bytes=12345, retries={"generate": 1})
    atomic(run / "state.json", encoded(prior))
    launches = []

    def completed_probe(_run, stage, _config, state, _lease):
        launches.append(stage)
        assert state["probe_new_games"] == 1
        atomic(
            run / "generation-probe.json",
            encoded(
                {
                    "run_sha256": digest(run / "run.json"),
                    "result": {"status": "paused", "games": 212, "deferred": 1},
                }
            ),
        )
        return 0, None

    monkeypatch.setattr(runner, "_run_stage", completed_probe)
    result = runner.work(run, pause_after_new_games=1)
    assert result["status"] == "ready_for_luna"
    assert result["generation"]["deferred"] == 1
    assert result["began_at"] == prior["began_at"]
    assert result["initial_swap_bytes"] == 12345
    assert result["retries"] == {"generate": 1}
    assert launches == ["generate"]
    assert not (run / "generate-complete.json").exists()
    assert not (run / "fit").exists()


def test_stage_probe_does_not_publish_false_generation_completion(prepared, monkeypatch):
    _, run, _, _, _ = prepared
    monkeypatch.setattr(runner, "_register_stage_group", lambda *_a: {})
    called = []

    def paused(_root, _output, _generation, *, pause_after_new_games):
        called.append(pause_after_new_games)
        return {"status": "paused", "games": 1, "deferred": 2}

    monkeypatch.setattr(runner, "generate", paused)
    result = runner.stage_run(run, "generate", pause_after_new_games=1)
    assert result["status"] == "paused"
    assert called == [1]
    assert not (run / "generate-complete.json").exists()
    assert runner._json(run / "generation-probe.json")["result"] == result


def test_probe_does_not_swallow_integrity_failure(prepared, monkeypatch):
    _, run, _, _, _ = prepared
    monkeypatch.setattr(runner, "_swap_bytes", lambda: 0)
    monkeypatch.setattr(runner, "_run_stage", lambda *_a: (1, None))
    result = runner.work(run, pause_after_new_games=1)
    assert result["status"] == "needs_astra"
    assert result["reason"] == "stage_exit_1"


def test_manifest_supplement_preserves_first_version_and_is_finite(prepared, monkeypatch):
    from open_shogi_training import evaluator_coverage

    _, run, config, _, _ = prepared
    data = run / "data"
    calls = []

    def preparation(*_args):
        value = {"version": 1 if not calls else 2}
        atomic(data / "dataset/manifest.json", encoded(value))
        atomic(data / "dataset-generated/manifest.json", encoded(value))
        return value

    monkeypatch.setattr(runner, "prepare", preparation)
    monkeypatch.setattr(
        evaluator_coverage,
        "coverage_report",
        lambda *_a: {"passed": False, "reasons": ["unique_new_train"]},
    )

    def supplement(_root, _data, _generation, *, supplemental):
        assert supplemental is True
        calls.append("supplement")
        return {"status": "complete"}

    monkeypatch.setattr(runner, "generate", supplement)
    monkeypatch.setattr(runner, "_dataset", lambda *_a: {"version": 2})
    result = runner._prepare_recovery(run, config)
    assert result == {"version": 2}
    assert calls == ["supplement"]
    archives = list((data / "manifest-attempts").glob("*/dataset/manifest.json"))
    assert len(archives) == 1
    assert json.loads(archives[0].read_text()) == {"version": 1}
    assert json.loads((data / "dataset/manifest.json").read_text()) == {"version": 2}


def test_manifest_coverage_pass_with_deferred_needs_no_supplement(prepared, monkeypatch):
    from open_shogi_training import evaluator_coverage

    _, run, config, _, _ = prepared
    monkeypatch.setattr(runner, "prepare", lambda *_a: {"accepted": True})
    monkeypatch.setattr(
        evaluator_coverage,
        "coverage_report",
        lambda *_a: {"passed": True, "deferred_roots": 1, "reasons": []},
    )
    monkeypatch.setattr(runner, "_dataset", lambda *_a: {"accepted": True})

    def forbidden(*_a, **_kw):
        pytest.fail("qualified manifest must progress without waiting for zero deferred")

    monkeypatch.setattr(runner, "generate", forbidden)
    assert runner._prepare_recovery(run, config) == {"accepted": True}


@pytest.mark.parametrize("interrupted_name", ["dataset-generated", "dataset"])
def test_manifest_archive_resume_after_each_committed_rename(
    prepared, monkeypatch, interrupted_name
):
    from open_shogi_training import evaluator_coverage

    _, run, config, _, _ = prepared
    data = run / "data"
    calls = []

    def preparation(*_args):
        value = {"version": 2 if calls else 1}
        for name in ("dataset", "dataset-generated"):
            atomic(data / name / "manifest.json", encoded(value))
        return value

    monkeypatch.setattr(runner, "prepare", preparation)
    monkeypatch.setattr(
        evaluator_coverage,
        "coverage_report",
        lambda *_a: {
            "passed": False,
            "reasons": ["unique_new_train"],
        },
    )
    monkeypatch.setattr(runner, "_dataset", lambda *_a: {"version": 2})

    def supplement(*_a, **_kw):
        calls.append(1)
        return {"status": "complete"}

    monkeypatch.setattr(runner, "generate", supplement)
    rename = Path.rename

    def interrupted(path, target):
        result = rename(path, target)
        if path == data / interrupted_name:
            raise OSError("injected interruption after committed rename")
        return result

    monkeypatch.setattr(Path, "rename", interrupted)
    with pytest.raises(OSError, match="injected interruption"):
        runner._prepare_recovery(run, config)
    monkeypatch.setattr(Path, "rename", rename)
    assert runner._prepare_recovery(run, config) == {"version": 2}
    assert len(calls) == 1
    assert not (data / "manifest-supplement.json").exists()
    archives = list((data / "manifest-attempts").glob("*/dataset/manifest.json"))
    assert len(archives) == 1
    assert json.loads(archives[0].read_text()) == {"version": 1}


def test_generation_progress_uses_accepted_tasks_not_retry_heartbeats(prepared):
    from open_shogi_training.evaluator_ledger import Ledger

    _, run, _, _, _ = prepared
    ledger = Ledger(run / "data")
    key, _ = ledger.task({"game": 1, "ply": 1})
    before = runner._progress_signature(run, "generate")
    ledger.begin(key, 2_000_000)
    assert runner._progress_signature(run, "generate") == before
    ledger.finish(key, "accepted", result={"quality": "exact"})
    assert runner._progress_signature(run, "generate") != before
    ledger.close()


def test_recovery_execution_requires_committed_migration(prepared):
    _, run, config, _, _ = prepared
    config["generation"]["recovery_policy"] = {"enabled": True}
    config["recovery_from"] = {"run_id": "parent"}
    with pytest.raises(FileNotFoundError):
        runner._require_migration(run, config)
    atomic(run / "data/inherited.json", encoded({"games": [0, 1]}))
    atomic(
        run / "migration.json",
        encoded(
            {
                "run_sha256": digest(run / "run.json"),
                "inherited_sha256": digest(run / "data/inherited.json"),
            }
        ),
    )
    runner._require_migration(run, config)
    atomic(run / "data/inherited.json", encoded({"games": []}))
    with pytest.raises(ValueError, match="artifact changed"):
        runner._require_migration(run, config)


def test_explicit_resource_epoch_preserves_deadline_and_attempts():
    prior = {"initial_swap_bytes": 10, "began_at": 100, "retries": {"generate": 1}}
    epoch = {
        "authorization": "explicit_user_approval_20260912",
        "original_initial_swap_bytes": 10,
        "initial_swap_bytes": 20,
    }
    result = runner._apply_resource_epoch(prior, epoch)
    assert result["initial_swap_bytes"] == 20
    assert result["original_initial_swap_bytes"] == 10
    assert result["began_at"] == 100 and result["retries"] == {"generate": 1}
    assert prior["initial_swap_bytes"] == 10
    with pytest.raises(ValueError, match="approved parent"):
        runner._apply_resource_epoch(prior, {**epoch, "original_initial_swap_bytes": 11})


def test_memory_pressure_measurement_fails_closed(monkeypatch):
    monkeypatch.setattr(
        runner.subprocess, "check_output", lambda *a, **k: "System-wide memory free percentage: 80%"
    )
    assert runner._memory_free_percent() == 80
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: "unavailable")
    with pytest.raises(ValueError, match="unavailable"):
        runner._memory_free_percent()


def test_memory_pressure_gate_stops_owned_stage(prepared, monkeypatch):
    _, run, config, _, _ = prepared
    config["resources"]["minimum_memory_free_percent"] = 50
    process = FakeProcess()
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(runner, "read_process_identity", lambda pid: "test-identity")
    monkeypatch.setattr(
        runner,
        "_stage_group_snapshot",
        lambda *a: {process.pid: (os.getpid(), 10, "S", process.pid)},
    )
    monkeypatch.setattr(runner, "_sample_owned", lambda *a: (10, {}))
    monkeypatch.setattr(runner, "_swap_bytes", lambda: 0)
    monkeypatch.setattr(runner, "_memory_free_percent", lambda: 49)
    monkeypatch.setattr(runner, "_cleanup", lambda *a, **k: {"remaining_processes": {}})
    with runner._lease(run) as lease:
        _, failure = runner._run_stage(run, "generate", config, stage_state(), lease)
    assert failure == "memory_pressure_limit"
    assert (run / "STOP").exists()
