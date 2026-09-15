"""Injected host observations never consume RAM or change the live recovery run."""

from __future__ import annotations

import copy
import json
import os
import signal
from types import SimpleNamespace

import pytest
from open_shogi_training import evaluator_run as runner
from open_shogi_training.evaluator_data import atomic, encoded
from test_evaluator_resume import resumable  # noqa: F401
from test_evaluator_run import prepared  # noqa: F401


@pytest.fixture
def resource_config():
    return {
        "resources": {
            "maximum_wall_seconds": 259200,
            "free_space_floor_gib": 60,
            "maximum_process_rss_gib": 12,
            "maximum_swap_growth_gib": 0.5,
            "minimum_memory_free_percent": 50,
        },
        "_resource_policy": copy.deepcopy(runner.RESOURCE_POLICY),
    }


def sample(at=1000.0, **changes):
    return {
        "at": at,
        "monotonic": at,
        "boot": "fixture-boot",
        "page_size": 16384,
        "swapouts": 100000,
        "swap_bytes": 17997758464,
        "rss_bytes": 2 * 1024**3,
        "free_bytes": 80 * 1024**3,
        "memory_free_percent": 60,
        "pressure": 1,
        **changes,
    }


def test_historical_swap_difference_is_not_current_pressure(resource_config, monkeypatch):
    monkeypatch.setattr(runner.time, "time", lambda: 1015.0)
    old = sample(1000)
    current = sample(1015)
    resource_config["initial_swap_bytes"] = 4191221186
    assert current["swap_bytes"] - resource_config["initial_swap_bytes"] > 12 * 1024**3
    assert runner._resource_condition(current, old, resource_config, admission=True) == "safe"


@pytest.mark.parametrize(
    "changes",
    [
        {"pressure": 4},
        {"memory_free_percent": 10},
        {"free_bytes": 59 * 1024**3},
        {"rss_bytes": 13 * 1024**3},
    ],
)
def test_immediate_danger_cannot_be_hidden_by_new_baseline(resource_config, monkeypatch, changes):
    monkeypatch.setattr(runner.time, "time", lambda: 1015.0)
    current = sample(1015, **changes)
    assert runner._resource_condition(current, sample(1000), resource_config).startswith(
        "critical:"
    )
    assert runner._resource_condition(current, None, resource_config, admission=True) != "safe"


def test_swapout_rate_uses_actual_page_size(resource_config, monkeypatch):
    monkeypatch.setattr(runner.time, "time", lambda: 1015.0)
    # 8,192 pages in 15 seconds is 8.53 MiB/s with 16 KiB pages,
    # but would be misclassified as 2.13 MiB/s with a fixed 4 KiB page.
    previous = sample(1000)
    current = sample(1015, swapouts=previous["swapouts"] + 8192)
    assert runner._resource_condition(current, previous, resource_config) != "safe"
    quiet = sample(1030, swapouts=current["swapouts"])
    monkeypatch.setattr(runner.time, "time", lambda: 1030.0)
    assert runner._resource_condition(quiet, current, resource_config) == "safe"


@pytest.mark.parametrize("changes", [{"boot": "next-boot"}, {"swapouts": 10}, {"monotonic": 999}])
def test_counter_discontinuity_requires_new_observations(resource_config, monkeypatch, changes):
    monkeypatch.setattr(runner.time, "time", lambda: 1015.0)
    assert (
        runner._resource_condition(
            sample(1015, **changes), sample(1000), resource_config, admission=True
        )
        != "safe"
    )


def test_stale_sample_is_not_safe(resource_config, monkeypatch):
    monkeypatch.setattr(runner.time, "time", lambda: 1100.0)
    assert runner._resource_condition(sample(1015), sample(1000), resource_config) != "safe"


@pytest.mark.parametrize("malformed", [False, True])
def test_read_only_sources_keep_gauges_and_page_counters_distinct(monkeypatch, tmp_path, malformed):
    replies = {
        ("vm_stat",): "unavailable"
        if malformed
        else ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\nSwapouts: 123456.\n"),
        ("sysctl", "-n", "kern.boottime"): "{ sec = 1234, usec = 567 }",
        ("sysctl", "-n", "kern.memorystatus_vm_pressure_level"): "1\n",
        ("sysctl", "-n", "vm.swapusage"): "total = 20000.00M used = 17164.00M free = 2836.00M",
        ("memory_pressure", "-Q"): "System-wide memory free percentage: 60%\n",
    }
    monkeypatch.setattr(
        runner.subprocess, "check_output", lambda args, **kwargs: replies[tuple(args)]
    )
    monkeypatch.setattr(
        runner.shutil, "disk_usage", lambda path: SimpleNamespace(free=80 * 1024**3)
    )
    if malformed:
        with pytest.raises(ValueError, match="measurement unavailable"):
            runner._resource_sample(tmp_path, 123)
    else:
        observed = runner._resource_sample(tmp_path, 123)
        assert observed["page_size"] == 16384
        assert observed["swapouts"] == 123456
        assert observed["swap_bytes"] == 17164 * 1024**2
        assert observed["rss_bytes"] == 123
        assert observed["boot"] == replies[("sysctl", "-n", "kern.boottime")]
        assert observed["source"]
        assert observed["units"]


def swap_failure(run):
    state = runner._state(run)
    state.update(
        status="needs_astra",
        reason="swap_limit",
        stage="generate",
        execution_attempt=4,
        cleanup={"remaining_processes": {}, "inventory_errors": []},
    )
    atomic(run / "attempts/000004-result.json", encoded(state))
    return state


def test_swap_recovery_requires_matching_attempt_result(resumable):  # noqa: F811
    _, run, _, _, _ = resumable
    state = swap_failure(run)
    assert runner._swap_recoverable(run, state)
    changed = {**state, "initial_swap_bytes": state["initial_swap_bytes"] + 1}
    assert not runner._swap_recoverable(run, changed)
    assert runner._swap_recoverable(run, state)


@pytest.mark.parametrize(
    "changes",
    [
        {"reason": "teacher_quality_failure"},
        {"reason": "attempt_result_write_failed"},
        {"stage": "train"},
        {"errors": [{"phase": "cleanup", "error": "fixture"}]},
        {"cleanup": {"remaining_processes": {"42": "fixture"}}},
        {"cleanup": {"remaining_processes": {}, "inventory_errors": ["fixture"]}},
    ],
)
def test_other_faults_are_not_swap_recovery(resumable, changes):  # noqa: F811
    _, run, _, _, _ = resumable
    state = {**swap_failure(run), **changes}
    atomic(run / "attempts/000004-result.json", encoded(state))
    assert not runner._swap_recoverable(run, state)


def admission_clock(monkeypatch, run, observations):
    now, calls = [1000.0], []
    monkeypatch.setattr(runner, "ROOT", run.parent)
    monkeypatch.setattr(runner.time, "time", lambda: now[0])
    monkeypatch.setattr(runner.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr(
        runner, "_process_table", lambda: {os.getpid(): (1, 2 * 1024**3, "S", os.getpid())}
    )
    values = iter(observations)

    def observe(path, rss):
        assert path == run
        assert rss == 2 * 1024**3
        calls.append(now[0])
        value = next(values)
        if isinstance(value, Exception):
            raise value
        return sample(now[0], **value)

    monkeypatch.setattr(runner, "_resource_sample", observe)
    return now, calls


def test_admission_requires_three_consecutive_quiet_intervals(
    resource_config, monkeypatch, tmp_path
):
    # A brief warning resets readiness, then three quiet intervals permit launch.
    _, calls = admission_clock(monkeypatch, tmp_path, [{}, {"pressure": 2}, {}, {}, {}])
    state = {"began_at": 900, "initial_swap_bytes": 4191221186, "execution_attempt": 4}
    before = copy.deepcopy(state)
    result = runner._resource_admission(tmp_path, resource_config, state)
    assert result["safe"]
    assert len(calls) == 5
    assert result["wait_seconds"] == 60
    assert result["previous_attempt"] == 4
    assert state == before


@pytest.mark.parametrize(
    "observations, expected_count",
    [
        ([{"pressure": 2}] * 5, 5),
        ([{}, *[{"swapouts": 100000 + i * 9000} for i in range(1, 5)]], 5),
        ([{"pressure": 4}], 1),
        ([OSError("fixture unavailable")], 1),
    ],
)
def test_unsafe_or_unmeasurable_admission_is_finite(
    resource_config, monkeypatch, tmp_path, observations, expected_count
):
    _, calls = admission_clock(monkeypatch, tmp_path, observations)
    result = runner._resource_admission(tmp_path, resource_config, {"began_at": 900})
    assert not result["safe"]
    assert len(calls) == expected_count
    assert result["wait_seconds"] <= 60
    saved = json.loads((tmp_path / "resource-admission.jsonl").read_text())
    assert saved == result


def test_admission_does_not_extend_absolute_deadline(resource_config, monkeypatch, tmp_path):
    resource_config["resources"]["maximum_wall_seconds"] = 20
    _, calls = admission_clock(monkeypatch, tmp_path, [{"pressure": 2}] * 5)
    state = {"began_at": 990, "retries": {"generate": 1}}
    before = copy.deepcopy(state)
    result = runner._resource_admission(tmp_path, resource_config, state)
    assert not result["safe"]
    assert result["reason"] == "wall_limit"
    assert len(calls) == 1
    assert state == before


def test_pause_during_last_admission_sample_prevents_launch(resource_config, monkeypatch, tmp_path):
    _, calls = admission_clock(monkeypatch, tmp_path, [{}, {}, {}, {}])
    observe = runner._resource_sample

    def pause_at_last_sample(*args):
        result = observe(*args)
        if len(calls) == 4:
            runner.stop(tmp_path)
        return result

    monkeypatch.setattr(runner, "_resource_sample", pause_at_last_sample)
    result = runner._resource_admission(tmp_path, resource_config, {"began_at": 900})
    assert not result["safe"]
    assert result["reason"] == "requested_stop"
    assert len(calls) == 4
    assert (tmp_path / "STOP").exists()


def test_resource_wait_resume_preserves_deadline_and_attempt_baseline(
    resumable,  # noqa: F811
    monkeypatch,
    resource_config,
):
    _, run, _, _, _ = resumable
    runner._approve_operations(run)
    original_verify = runner.verify

    def verify_with_policy(*args, **kwargs):
        config = original_verify(*args, **kwargs)
        config["_resource_policy"] = resource_config["_resource_policy"]
        return config

    monkeypatch.setattr(runner, "verify", verify_with_policy)
    before = runner._state(run)
    blocked = {"safe": False, "reason": "memory_pressure", "samples": [], "wait_seconds": 60}
    monkeypatch.setattr(runner, "_resource_admission", lambda *args: blocked)
    with runner._lease(run):
        _, result = runner._prepare_resume(run)
    assert result["status"] == "stopped"
    assert result["reason"] == "resource_wait"
    assert not list((run / "attempts").glob("*.json"))
    assert result["began_at"] == before["began_at"]
    safe = {"safe": True, "reason": "safe", "samples": [sample()], "wait_seconds": 45}
    monkeypatch.setattr(runner, "_resource_admission", lambda *args: safe)
    with runner._lease(run):
        _, resumed = runner._prepare_resume(run)
    assert resumed["execution_attempt"] == 1
    assert resumed["initial_swap_bytes"] == before["initial_swap_bytes"]
    assert resumed["began_at"] == before["began_at"]
    assert resumed["retries"] == before["retries"]
    record = json.loads((run / "attempts/000001.json").read_text())
    assert resumed["resource_baseline"] == record["resource_baseline"]
    assert record["resource_baseline"]["attempt"] == 1
    assert record["previous_state"]["reason"] == "resource_wait"


@pytest.mark.parametrize(
    "observations, expected, interruption",
    [
        ([{"pressure": 2}, {}, {}], None, None),
        ([{"pressure": 2}, {"pressure": 2}], "resource_wait:memory_pressure", None),
        ([{"pressure": 4}], "resource_wait:critical:memory_pressure", None),
        ([OSError("fixture monitor unavailable")], "resource_wait:measurement_unavailable:", None),
        (
            [{"swapouts": 109000}, {"swapouts": 118000}],
            "resource_wait:swap_activity",
            None,
        ),
        ([{"pressure": 4}], "resource_wait:critical:memory_pressure", "owned"),
        ([{"pressure": 4}], "stage_exit_-15_during_stop", "unowned"),
        ([{"pressure": 4}], "interrupted_snapshot_failed", "bad_cursor"),
    ],
)
@pytest.mark.parametrize("prior_history", ["same_attempt", "previous_boot"])
def test_runtime_resource_policy_noise_sustained_danger_and_measurement_failure(
    prepared,  # noqa: F811
    monkeypatch,
    resource_config,
    observations,
    expected,
    interruption,
    prior_history,
):
    _, run, config, _, _ = prepared
    config.update(resource_config)
    config["resources"]["stalled_seconds"] = 1800
    now, samples, launches, cleanup_grace = [1000.0], [{}, *observations], [], []
    baseline = {**sample(900), "attempt": 1}
    state = {
        "began_at": 900,
        "initial_swap_bytes": 4191221186,
        "retries": {},
        "execution_attempt": 1,
        "resource_baseline": baseline,
        # A later stage uses the recent history; baseline remains 115 seconds old.
        "resource_history": {"last": sample(1000)},
    }
    if prior_history == "previous_boot":
        baseline.update(sample(1000))
        state["resource_history"]["last"] = sample(5000, boot="previous-boot")
    atomic(run / "attempts/000001.json", encoded({"resource_baseline": baseline}))
    process = SimpleNamespace(pid=900001, returncode=None)
    monkeypatch.setattr(runner.time, "time", lambda: now[0])
    monkeypatch.setattr(runner.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(runner, "_process_table", lambda: {os.getpid(): (1, 1024, "S", 1)})

    def observe(*args):
        now[0] += 15
        value = samples.pop(0)
        if isinstance(value, Exception):
            raise value
        return sample(now[0], **value)

    def launch(*args):
        launches.append(1)
        return process

    def cleanup(*args, **kwargs):
        cleanup_grace.append(kwargs["grace_seconds"])
        process.returncode = (
            -signal.SIGTERM
            if interruption
            else runner.PAUSED_EXIT
            if (run / "STOP").exists()
            else 0
        )
        return {
            "remaining_processes": {},
            "inventory_errors": [],
            "signalled_pids": [process.pid] if interruption in ("owned", "bad_cursor") else [],
        }

    def interrupted_snapshot(*args):
        if interruption == "bad_cursor":
            raise ValueError("fixture corrupt cursor")
        return {"games": 1, "rows": 12, "tasks": {"running": 1}}

    monkeypatch.setattr(runner, "_resource_sample", observe)
    monkeypatch.setattr(runner, "_launch", launch)
    monkeypatch.setattr(runner, "read_process_identity", lambda pid: "fixture-identity")
    monkeypatch.setattr(
        runner,
        "_stage_group_snapshot",
        lambda *args: {process.pid: (1, 1024, "S" if samples else "Z", process.pid)},
    )
    monkeypatch.setattr(runner, "_sample_owned", lambda *args: (1024, {}))
    monkeypatch.setattr(runner, "_progress_signature", lambda *args: (now[0],))
    monkeypatch.setattr(runner, "_cleanup", cleanup)
    monkeypatch.setattr(runner, "_mark_group_cleaned", lambda *args: None)
    monkeypatch.setattr(runner, "_resume_snapshot", interrupted_snapshot)
    with runner._lease(run) as lease:
        code, failure = runner._run_stage(run, "generate", config, state, lease)
    assert launches == [1]
    assert state["resource_baseline"] == baseline
    if expected is None:
        assert (code, failure) == (0, None)
        assert not (run / "STOP").exists()
    else:
        assert code == (-signal.SIGTERM if interruption else runner.PAUSED_EXIT)
        assert failure.startswith(expected)
        assert (run / "STOP").exists()
        critical = isinstance(observations[0], dict) and observations[0].get("pressure") == 4
        assert cleanup_grace == [
            0 if critical else runner.RESOURCE_POLICY["cooperative_stop_seconds"]
        ]


@pytest.mark.parametrize(
    "failure, expected_status",
    [
        ("resource_wait:swap_activity", "stopped"),
        ("resource_wait:measurement_unavailable:fixture", "stopped"),
        ("requested_stop", "stopped"),
        ("teacher_quality_failure", "needs_astra"),
        ("stage_exit_1_during_stop", "needs_astra"),
    ],
)
def test_work_keeps_resource_wait_distinct_from_faults(
    prepared,  # noqa: F811
    monkeypatch,
    failure,
    expected_status,
):
    _, run, _, _, _ = prepared
    monkeypatch.setattr(runner, "_swap_bytes", lambda: 0)
    monkeypatch.setattr(runner, "_run_stage", lambda *args: (runner.PAUSED_EXIT, failure))
    result = runner.work(run)
    assert result["status"] == expected_status
    assert result["reason"] == failure


def test_stale_running_requires_no_writer_and_explicit_resume(resumable, monkeypatch):  # noqa: F811
    _, run, _, _, _ = resumable
    runner._approve_operations(run)
    with runner._lease(run):
        _, state = runner._prepare_resume(run)
    state.update(status="running", reason="interrupted", pid=999999, process_identity="old")
    atomic(run / "state.json", encoded(state))
    monkeypatch.setattr(runner, "read_process_identity", lambda pid: "old")
    with runner._lease(run), pytest.raises(ValueError, match="live process"):
        runner._prepare_resume(run)
    monkeypatch.setattr(runner, "read_process_identity", lambda pid: None)
    with runner._lease(run):
        _, resumed = runner._prepare_resume(run)
    assert resumed["execution_attempt"] == 2
    assert resumed["began_at"] == state["began_at"]
    assert resumed["retries"] == state["retries"]


def test_same_attempt_baseline_change_is_rejected(resumable, monkeypatch):  # noqa: F811
    _, run, _, _, _ = resumable
    runner._approve_operations(run)
    with runner._lease(run):
        _, state = runner._prepare_resume(run)
    state["resource_baseline"] = {**sample(), "attempt": 1}
    atomic(run / "state.json", encoded(state))
    original = runner.verify

    def with_policy(*args, **kwargs):
        return {**original(*args, **kwargs), "_resource_policy": runner.RESOURCE_POLICY}

    monkeypatch.setattr(runner, "verify", with_policy)
    monkeypatch.setattr(runner, "_run_stage", lambda *args: pytest.fail("rebased attempt launched"))
    result = runner.work(run)
    assert result["status"] == "needs_astra"
    assert "baseline changed" in result["reason"]


def test_outer_resume_interrupt_requests_stop(resumable, monkeypatch):  # noqa: F811
    _, run, _, _, _ = resumable
    runner._approve_operations(run)

    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "_launch", interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner.start(run, recover_startup=True)
    assert (run / "STOP").exists()
