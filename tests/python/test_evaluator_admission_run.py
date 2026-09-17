"""Explicit dataset revision closes only the reviewed failure and preserves evidence."""

import json

import pytest
from open_shogi_training import evaluator_run as runner
from open_shogi_training.evaluator_data import atomic, encoded
from test_evaluator_resource import resource_config, sample  # noqa: F401
from test_evaluator_resume import resumable  # noqa: F401
from test_evaluator_run import prepared  # noqa: F401


def focus_failure(run):
    config = runner.verify(run)
    snapshot = runner._resume_snapshot(run, config)
    state = runner._state(run)
    state.update(
        status="needs_astra",
        stage="generate",
        reason="stage_exit_1",
        execution_attempt=11,
        cleanup={"remaining_processes": {}},
    )
    atomic(run / "state.json", encoded(state))
    atomic(run / "attempts/000011.json", encoded({"snapshot": snapshot}))
    atomic(run / "attempts/000011-result.json", encoded(state))
    atomic(run / "data/recovery-coverage.json", encoded({"reasons": ["focus_total"]}))
    atomic(
        run / "data/generation-progress.json", encoded({"games": 1, "status": "coverage_exhausted"})
    )
    (run / "generate-1.log").write_text(
        "ValueError: finite generation coverage exhausted: coverage_exhausted\n"
    )
    return state


def test_admission_preserves_failure_seal_budgets_and_is_idempotent(resumable, monkeypatch):  # noqa: F811
    run = resumable[1]
    monkeypatch.setitem(runner.ADMISSION_POLICY, "run_id", "contract-test")
    before = focus_failure(run)
    originals = {
        name: (run / name).read_bytes()
        for name in ("run.json", "seal.json", "state.json", "data/tasks.sqlite3")
    }
    ref = runner._approve_operations(run, admit_existing_data=True)
    assert runner._approve_operations(run, admit_existing_data=True) == ref
    assert runner._resume_idle_state(run) == before
    revision = json.loads(runner.inside(ref["path"]).read_text())
    config = runner.verify(run, operation_revision=revision)
    assert "_dataset_admission" in config
    assert {name: (run / name).read_bytes() for name in originals} == originals
    manifest = json.loads((run / "data/admission.json").read_text())
    assert manifest["consumption"]["hard_attempts"] == 360
    assert manifest["generation_budget_closed"]


@pytest.mark.parametrize("failure", ["nan", "other_coverage", "live"])
def test_admission_cannot_clear_unrelated_failure(resumable, monkeypatch, failure):  # noqa: F811
    run = resumable[1]
    monkeypatch.setitem(runner.ADMISSION_POLICY, "run_id", "contract-test")
    state = focus_failure(run)
    if failure == "nan":
        (run / "generate-1.log").write_text("FloatingPointError: nonfinite\n")
    elif failure == "other_coverage":
        atomic(
            run / "data/recovery-coverage.json",
            encoded({"reasons": ["focus_total", "family_completed:bad"]}),
        )
    else:
        monkeypatch.setattr(runner, "_residual_stage_group", lambda _: {"pid": 123})
    with pytest.raises(ValueError):
        runner._approve_operations(run, admit_existing_data=True)
    assert runner._state(run) == state
    assert not (run / "approved-operation.json").exists()


def test_host_pressure_is_diagnostic_but_disk_is_required(resource_config):  # noqa: F811
    resource_config["_dataset_admission"] = {"policy": "itemwise-focus-v1"}
    observed = sample(
        pressure=4,
        memory_free_percent=0,
        rss_bytes=100 * 1024**3,
        swap_bytes=200 * 1024**3,
        swapouts=10**12,
    )
    assert runner._resource_condition(observed, None, resource_config, admission=True) == "safe"
    observed["free_bytes"] = 59 * 1024**3
    assert runner._resource_condition(observed, None, resource_config) == "critical:disk"


def test_training_completion_uses_same_admission_identity(resumable):  # noqa: F811
    run, config = resumable[1:3]
    ref = runner._approve_operations(run)
    state = runner._state(run)
    state["operation_revision"] = ref
    atomic(run / "state.json", encoded(state))
    config["_dataset_admission"] = {"sha256": "a" * 64}
    atomic(run / "data/dataset/manifest.json", encoded({"fixture": True}))
    atomic(run / "fit/best.osaval03", b"fixture model bytes")
    atomic(run / "fit/checkpoint-000001.pt", b"fixture checkpoint")
    atomic(
        run / "fit/resume.json",
        encoded(
            {
                "step": 1,
                "path": "checkpoint-000001.pt",
                "sha256": runner.digest(run / "fit/checkpoint-000001.pt"),
            }
        ),
    )
    result = {
        "status": "complete",
        "step": 1,
        "identity": runner._training_identity(run, config),
        "best_sha256": runner.digest(run / "fit/best.osaval03"),
    }
    atomic(run / "fit/training.json", encoded(result))
    assert runner._training_summary(run, config) == result
    result["identity"].pop("dataset_admission")
    atomic(run / "fit/training.json", encoded(result))
    with pytest.raises(ValueError, match="identity mismatch"):
        runner._training_summary(run, config)


@pytest.mark.parametrize("failures", [1, 4])
def test_allocation_wait_keeps_budget_and_pause_across_resumes(resumable, monkeypatch, failures):  # noqa: F811
    run = resumable[1]
    config = {"_dataset_admission": {}, "resources": {"free_space_floor_gib": 60}}
    atomic(run / "allocation-retries.json", encoded({"train:0": failures}))
    atomic(
        run / "allocation-failure.json",
        encoded({"at": 0, "resource_sample": {"memory_free_percent": 5, "pressure": 4}}),
    )
    observations = iter(
        [OSError("measurement unavailable"), sample(pressure=1, memory_free_percent=60)]
    )
    monkeypatch.setattr(runner.time, "time", lambda: 100)
    monkeypatch.setattr(runner, "_process_table", lambda: {runner.os.getpid(): (0, 0)})
    count = 0

    def wait(_):
        nonlocal count
        count += 1
        if count == 3:
            (run / "STOP").touch()

    def measurement(*_):
        value = next(observations, sample(pressure=1, memory_free_percent=60))
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(runner.time, "sleep", wait)
    monkeypatch.setattr(runner, "_resource_sample", measurement)
    assert runner._wait_allocation(run, config, runner._state(run), "train") is (failures == 1)
    assert json.loads((run / "allocation-retries.json").read_text()) == {"train:0": failures}
    if failures == 4:
        assert not runner._wait_allocation(run, config, runner._state(run), "train")


def test_only_supervised_prepare_interruption_is_recoverable(resumable):  # noqa: F811
    run = resumable[1]
    state = runner._state(run)
    state.update(
        stage="prepare",
        reason="stage_exit_-15_during_stop",
        execution_attempt=1,
        stage_pid=123,
        cleanup={"signalled_pids": [123], "remaining_processes": {}},
    )
    (run / "STOP").touch()
    atomic(run / "attempts/000001-result.json", encoded(state))
    assert runner._startup_recoverable(run, state)
    state["stage"] = "train"
    assert not runner._startup_recoverable(run, state)
    state["stage"] = "prepare"
    state["cleanup"]["signalled_pids"] = []
    assert not runner._startup_recoverable(run, state)
