"""Calendar removal changes reviewed operations, never sealed science or task budgets."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
from open_shogi_training import evaluator_run as runner
from open_shogi_training.evaluator_data import atomic, encoded
from test_evaluator_resource import (  # noqa: F401
    admission_clock,
    resource_config,
    sample,
    swap_failure,
)
from test_evaluator_resume import resumable  # noqa: F401
from test_evaluator_run import prepared  # noqa: F401


@pytest.fixture
def calendar_run(resumable, monkeypatch):  # noqa: F811
    monkeypatch.setitem(runner.CALENDAR_POLICY, "run_id", "contract-test")
    return resumable[1]


def revision(run, ref):
    return json.loads(runner.inside(ref["path"]).read_text())


def test_calendar_approval_retry_preserves_originals_and_history(calendar_run, monkeypatch):
    run = calendar_run
    old = runner._approve_operations(run)
    names = ["run.json", "seal.json", "state.json", "data/tasks.sqlite3"]
    before = {name: (run / name).read_bytes() for name in names}
    original_atomic = runner.atomic

    def interrupted(path, content):
        if path == run / "approved-operation.json":
            raise OSError("fixture interrupted before pointer publication")
        return original_atomic(path, content)

    monkeypatch.setattr(runner, "atomic", interrupted)
    with pytest.raises(OSError, match="pointer publication"):
        runner._approve_operations(run, abolish_calendar_limit=True)
    assert json.loads((run / "approved-operation.json").read_text()) == old
    assert len(list((run / "operations").glob("*.json"))) == 2
    monkeypatch.setattr(runner, "atomic", original_atomic)
    approved = runner._approve_operations(run, abolish_calendar_limit=True)
    assert runner._approve_operations(run, abolish_calendar_limit=True) == approved
    assert runner._approve_operations(run) == approved
    assert len(list((run / "operations").glob("*.json"))) == 2
    saved = revision(run, approved)
    assert saved["supersedes"] == old
    assert saved["original_seal"] == runner._reference(run / "seal.json")
    assert saved["calendar_policy"] == runner.CALENDAR_POLICY
    assert {name: (run / name).read_bytes() for name in names} == before
    config = runner.verify(run, operation_revision=saved)
    assert config["_calendar_policy"] == runner.CALENDAR_POLICY
    assert config["resources"]["maximum_wall_seconds"] == saved["original_maximum_wall_seconds"]


def test_calendar_approval_cannot_apply_to_unapproved_run(resumable):  # noqa: F811
    run = resumable[1]
    with pytest.raises(ValueError, match="unauthorized run"):
        runner._approve_operations(run, abolish_calendar_limit=True)
    assert not (run / "approved-operation.json").exists()


def swap_ready(run, monkeypatch):
    snapshot = runner._resume_snapshot(run, runner.verify(run))
    state = swap_failure(run)
    atomic(run / "state.json", encoded(state))
    atomic(run / "attempts/000004.json", encoded({"snapshot": snapshot}))
    runner._approve_operations(run, abolish_calendar_limit=True)
    original_verify = runner.verify

    def verified_resource_policy(*args, **kwargs):
        config = original_verify(*args, **kwargs)
        config["_resource_policy"] = runner.RESOURCE_POLICY
        return config

    monkeypatch.setattr(runner, "verify", verified_resource_policy)
    return state


@pytest.mark.parametrize("now", [50.0, 1_000_000.0, 10_000_000.0])
def test_expired_swap_resume_keeps_history_and_budgets(calendar_run, monkeypatch, now):
    run = calendar_run
    before = swap_ready(run, monkeypatch)
    original = (run / "attempts/000004-result.json").read_bytes()
    monkeypatch.setattr(runner.time, "time", lambda: now)
    monkeypatch.setattr(
        runner,
        "_resource_admission",
        lambda *args: {
            "safe": True,
            "reason": "safe",
            "samples": [sample(now)],
            "wait_seconds": 45,
        },
    )
    with runner._lease(run):
        config, state = runner._prepare_resume(run)
    assert not runner._calendar_expired(config, before)
    assert state["execution_attempt"] == 5
    assert state["began_at"] == before["began_at"]
    assert state["retries"] == before["retries"]
    saved = json.loads((run / "attempts/000005.json").read_text())
    assert saved["previous_state"] == before
    assert saved["snapshot"]["counters"]["hard_attempts"] == 360
    assert saved["snapshot"]["tasks"] == {"deferred": 1}
    assert (run / "attempts/000004-result.json").read_bytes() == original
    # Child and supervisor resolve the reviewed policy from the persisted attempt.
    assert runner.verify(run)["_calendar_policy"] == runner.CALENDAR_POLICY


@pytest.mark.parametrize("blocker", ["danger", "data", "lease"])
def test_calendar_removal_does_not_clear_real_blockers(calendar_run, monkeypatch, blocker):
    run = calendar_run
    before = swap_ready(run, monkeypatch)
    monkeypatch.setattr(runner.time, "time", lambda: 1_000_000.0)
    if blocker == "danger":
        monkeypatch.setattr(
            runner,
            "_resource_admission",
            lambda *args: {
                "safe": False,
                "reason": "critical:memory_pressure",
                "samples": [],
                "wait_seconds": 0,
            },
        )
        with runner._lease(run):
            _, state = runner._prepare_resume(run)
        assert state["resume_blocked"]["reason"] == "critical:memory_pressure"
    elif blocker == "data":
        (run / "data/games/000022.json.gz").write_bytes(b"corrupt fixture")
        with runner._lease(run), pytest.raises(ValueError, match="artifact changed"):
            runner._prepare_resume(run)
    else:
        monkeypatch.setattr(runner, "_launch", lambda *args: pytest.fail("double launch"))
        with runner._lease(run):
            assert runner.start(run, recover_startup=True)["lease_held"]
    assert runner._state(run) == before
    assert not (run / "attempts/000005.json").exists()


@pytest.mark.parametrize("began", [0, 2_000_000])
def test_resource_admission_ignores_calendar_but_keeps_finite_sampling(
    calendar_run,
    resource_config,  # noqa: F811
    monkeypatch,
    began,
):
    run = calendar_run
    ref = runner._approve_operations(run, abolish_calendar_limit=True)
    config = runner.verify(run, operation_revision=revision(run, ref))
    config.update(resource_config)
    config["resources"]["maximum_wall_seconds"] = 20
    _, calls = admission_clock(monkeypatch, run, [{}, {}, {}, {}])
    result = runner._resource_admission(run, config, {"began_at": began})
    assert result["safe"]
    assert len(calls) == 4
    assert result["wait_seconds"] == 45


@pytest.mark.parametrize("stage", runner.STAGES)
def test_all_stage_launch_and_monitor_use_reviewed_calendar(
    calendar_run,
    resource_config,  # noqa: F811
    monkeypatch,
    stage,
):
    run = calendar_run
    ref = runner._approve_operations(run, abolish_calendar_limit=True)
    config = runner.verify(run, operation_revision=revision(run, ref))
    config.update(resource_config)
    config["resources"].update(stalled_seconds=1800, maximum_wall_seconds=20)
    now, observed, launches = [1000.0], [], []
    state = {
        "began_at": 0,
        "initial_swap_bytes": 0,
        "retries": {},
        "execution_attempt": 1,
        "resource_baseline": {**sample(1000), "attempt": 1},
    }
    process = SimpleNamespace(pid=900001, returncode=None)
    monkeypatch.setattr(runner.time, "time", lambda: now[0])
    monkeypatch.setattr(runner.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(runner, "_process_table", lambda: {os.getpid(): (1, 1024, "S", 1)})

    def observe(*args):
        now[0] += 15
        observed.append(now[0])
        return sample(now[0])

    def cleanup(*args, **kwargs):
        process.returncode = 0
        return {"remaining_processes": {}, "inventory_errors": []}

    def launch(*args):
        launches.append(stage)
        return process

    monkeypatch.setattr(runner, "_resource_sample", observe)
    monkeypatch.setattr(runner, "_launch", launch)
    monkeypatch.setattr(runner, "read_process_identity", lambda pid: "fixture")
    monkeypatch.setattr(
        runner,
        "_stage_group_snapshot",
        lambda *args: {process.pid: (1, 1024, "Z" if len(observed) == 2 else "S", process.pid)},
    )
    monkeypatch.setattr(runner, "_sample_owned", lambda *args: (1024, {}))
    monkeypatch.setattr(runner, "_progress_signature", lambda *args: (now[0],))
    monkeypatch.setattr(runner, "_cleanup", cleanup)
    monkeypatch.setattr(runner, "_mark_group_cleaned", lambda *args: None)
    with runner._lease(run) as lease:
        assert runner._run_stage(run, stage, config, state, lease) == (0, None)
    assert launches == [stage]
    assert len(observed) == 2


def test_work_accepts_clock_rollback_and_preserves_user_pause(calendar_run, monkeypatch):
    run = calendar_run
    runner._approve_operations(run, abolish_calendar_limit=True)
    with runner._lease(run):
        runner._prepare_resume(run)
    runner.stop(run)
    monkeypatch.setattr(runner.time, "time", lambda: 50.0)
    monkeypatch.setattr(runner, "_run_stage", lambda *args: pytest.fail("paused stage launched"))
    state = runner.work(run)
    assert state["reason"] == "requested_stop"
    assert state["began_at"] == 100
    assert (run / "STOP").exists()


def test_diagnose_separates_latest_rejection_and_collects_faults(calendar_run, monkeypatch):
    run = calendar_run
    swap_ready(run, monkeypatch)
    atomic(run / "last-resume.json", encoded({"reason": "wall_limit", "status": "blocked"}))
    (run / "data/games/000022.json.gz").write_bytes(b"corrupt fixture")
    monkeypatch.setattr(
        runner,
        "_resource_sample",
        lambda *args: (_ for _ in ()).throw(ValueError("fixture resource unavailable")),
    )
    report = runner.diagnose(run)
    assert report["previous_attempt"]["reason"] == "swap_limit"
    assert report["latest_resume"]["reason"] == "wall_limit"
    assert report["checks"]["data_receipts_ledger_cursor"]["status"] == "blocked"
    assert report["checks"]["resource_measurement"]["status"] == "blocked"
    assert report["checks"]["calendar"]["status"] == "pass"
    assert report["resume"] == "blocked"


@pytest.mark.parametrize(
    "outcome", ["invalid_data", "spawn_error", "bootstrap_exit", "timeout", "ack"]
)
def test_latest_resume_outcome_is_published_under_original_lease(
    calendar_run, monkeypatch, outcome
):
    run = calendar_run
    runner._approve_operations(run, abolish_calendar_limit=True)
    original_record = runner._record_resume
    records = []

    def record_while_exclusive(*args, **kwargs):
        # A competing resume cannot acquire the lease before this result is durable.
        with pytest.raises(BlockingIOError), runner._lease(run):
            pytest.fail("last-resume publication lost the original lease")
        records.append(args[1:3])
        original_record(*args, **kwargs)

    monkeypatch.setattr(runner, "_record_resume", record_while_exclusive)
    process = SimpleNamespace(pid=900001, returncode=1 if outcome == "bootstrap_exit" else None)
    process.poll = lambda: process.returncode

    def launch(*args):
        if outcome == "spawn_error":
            raise OSError("fixture EBADF before child")
        if outcome == "ack":
            state = runner._state(run)
            state["pid"] = process.pid
            atomic(run / "state.json", encoded(state))
        return process

    monkeypatch.setattr(runner, "_launch", launch)
    if outcome == "invalid_data":
        (run / "data/games/000022.json.gz").write_bytes(b"corrupt fixture")
        monkeypatch.setattr(runner, "_resource_sample", lambda *args: sample())
        with pytest.raises(ValueError, match="artifact changed"):
            runner.start(run, recover_startup=True)
        expected = "blocked"
    elif outcome == "spawn_error":
        with pytest.raises(OSError, match="EBADF"):
            runner.start(run, recover_startup=True)
        expected = "blocked"
    elif outcome == "timeout":
        ticks = iter([1000, 1031])
        monkeypatch.setattr(runner.time, "monotonic", lambda: next(ticks))
        with pytest.raises(RuntimeError, match="acknowledgement timed out"):
            runner.start(run, recover_startup=True)
        expected = "unconfirmed"
    else:
        runner.start(run, recover_startup=True)
        expected = "started" if outcome == "ack" else "blocked"
    saved = json.loads((run / "last-resume.json").read_text())
    assert saved["status"] == expected
    assert records[-1] == (saved["status"], saved["reason"])
    if outcome != "invalid_data":
        assert saved["execution_attempt"] == 1
    assert len(records) == 1


def test_diagnose_rejects_new_committed_file_outside_approved_inventory(calendar_run, monkeypatch):
    run = calendar_run
    runner._approve_operations(run, abolish_calendar_limit=True)
    original_code = runner._code_identity()
    monkeypatch.setattr(
        runner,
        "_code_identity",
        lambda: {
            **original_code,
            "files": {**original_code["files"], "training/new-committed.py": "a" * 64},
        },
    )
    monkeypatch.setattr(runner, "_resource_sample", lambda *args: sample(runner.time.time()))
    report = runner.diagnose(run)
    assert report["checks"]["approved_code_seal_inputs"]["status"] == "pass"
    assert report["checks"]["committed_code"]["status"] == "blocked"
    assert report["resume"] == "blocked"
    with runner._lease(run), pytest.raises(ValueError, match="code differs"):
        runner._prepare_resume(run)
