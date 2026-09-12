"""Operational resume preserves the sealed experiment and durable attempt history."""

from __future__ import annotations

import copy
import errno
import gzip
import hashlib
import json

import pytest
from open_shogi_training import evaluator_run as runner
from open_shogi_training.evaluator_data import atomic, digest, encoded
from open_shogi_training.evaluator_ledger import Ledger
from test_evaluator_run import prepared  # noqa: F401


@pytest.fixture
def resumable(prepared):  # noqa: F811
    root, run, config, source, receipt = prepared
    atomic(run / "data/generation.json", encoded(config["generation"]))
    raw = {"game": 22, "records": [{"fixture": "retained accepted data"}]}
    shard = run / "data/games/000022.json.gz"
    atomic(shard, gzip.compress(encoded(raw), mtime=0))
    atomic(
        shard.with_suffix(".receipt.json"),
        encoded({"game": 22, "rows": 1, "sha256": digest(shard)}),
    )
    ledger = Ledger(run / "data")
    key, _ = ledger.task({"game": 23, "branch": "root", "fixture": "previous D12 task"})
    with ledger.db:
        ledger.db.execute(
            "UPDATE tasks SET status='deferred',attempts=? WHERE id=?",
            (json.dumps([{"nodes": 2_000_000}, {"nodes": 32_000_000}]), key),
        )
        ledger.db.execute("INSERT INTO counters VALUES('hard_attempts',360)")
    ledger.close()
    state = runner._state(run)
    state.update(
        status="stopped",
        reason="requested_stop",
        began_at=100,
        initial_swap_bytes=1234,
        retries={"generate": 1},
    )
    atomic(run / "state.json", encoded(state))
    return root, run, config, source, receipt


def test_approved_revision_is_reused_and_preserves_prior_failure_and_budgets(resumable):
    _, run, _, _, _ = resumable
    state = runner._state(run)
    state.update(
        status="needs_astra",
        reason="supervisor_spawn_failed",
        startup_failure="before_spawn",
        errors=[{"phase": "supervisor_spawn", "traceback": "original EBADF traceback"}],
    )
    atomic(run / "state.json", encoded(state))
    seal_before = (run / "seal.json").read_bytes()
    contract_before = (run / "run.json").read_bytes()
    state_before = (run / "state.json").read_bytes()
    runner._approve_operations(run)
    assert (run / "state.json").read_bytes() == state_before
    with runner._lease(run):
        _, first = runner._prepare_resume(run)
    record = json.loads((run / "attempts/000001.json").read_text())
    assert record["previous_state"] == state
    assert record["snapshot"]["games"] == 1
    assert record["snapshot"]["rows"] == 1
    assert record["snapshot"]["tasks"] == {"deferred": 1}
    assert record["snapshot"]["counters"]["hard_attempts"] == 360
    assert first["status"] == "running"
    assert first["began_at"] == 100
    assert first["initial_swap_bytes"] == 1234
    assert first["retries"] == {"generate": 1}
    first.update(status="stopped", reason="requested_stop")
    atomic(run / "state.json", encoded(first))
    with runner._lease(run):
        _, second = runner._prepare_resume(run)
    assert second["execution_attempt"] == 2
    assert second["operation_revision"] == first["operation_revision"]
    assert len(list((run / "operations").glob("*.json"))) == 1
    assert json.loads((run / "attempts/000001.json").read_text()) == record
    assert (run / "seal.json").read_bytes() == seal_before
    assert (run / "run.json").read_bytes() == contract_before


@pytest.mark.parametrize("corruption", ["shard", "checkpoint"])
def test_resume_validation_rejects_bad_output_or_cursor_without_state_change(resumable, corruption):
    _, run, _, _, _ = resumable
    runner._approve_operations(run)
    if corruption == "shard":
        (run / "data/games/000022.json.gz").write_bytes(b"truncated output")
    else:
        ledger = Ledger(run / "data")
        ledger.checkpoint(23, {"ply": 2, "moves": [], "records": []})
        ledger.close()
    before = (run / "state.json").read_bytes()
    with runner._lease(run), pytest.raises(ValueError):
        runner._prepare_resume(run)
    assert (run / "state.json").read_bytes() == before
    assert not list((run / "attempts").glob("*.json"))


def test_approval_rejects_changed_non_operational_source(resumable, monkeypatch):
    root, run, config, _, _ = resumable
    code = copy.deepcopy(config["code"])
    changed = "training/runner-fixture.py"
    (root / changed).write_text("# changed teacher or learning code\n")
    code["files"][changed] = digest(root / changed)
    monkeypatch.setattr(runner, "_code_identity", lambda: code)
    before = (run / "state.json").read_bytes()
    with pytest.raises(ValueError, match="operational"):
        runner._approve_operations(run)
    assert (run / "state.json").read_bytes() == before
    assert not (run / "approved-operation.json").exists()


def test_resume_never_adopts_code_changed_after_approval(resumable):
    root, run, _, _, _ = resumable
    runner._approve_operations(run)
    before = (run / "state.json").read_bytes()
    approved = (run / "approved-operation.json").read_bytes()
    (root / "training/runner-fixture.py").write_text("# unapproved new revision\n")
    with runner._lease(run), pytest.raises(ValueError, match="artifact changed"):
        runner._prepare_resume(run)
    assert (run / "state.json").read_bytes() == before
    assert (run / "approved-operation.json").read_bytes() == approved
    assert len(list((run / "operations").glob("*.json"))) == 1


def test_operational_config_revision_cannot_change_training(resumable, monkeypatch):
    _, run, config, _, _ = resumable
    source_name = "configs/evaluator-main.json"
    original = encoded({"training": {"batch_size": 256}, "operations": {"old": True}})
    changed = encoded({"training": {"batch_size": 512}, "operations": {"resume": True}})
    config["code"]["files"][source_name] = hashlib.sha256(original).hexdigest()
    revision = {
        "schema": "open_shogiai_operation_revision/v1",
        "run_sha256": digest(run / "run.json"),
        "original_code_commit": config["code"]["commit"],
        "commit": "b" * 40,
        "files": {source_name: hashlib.sha256(changed).hexdigest()},
    }
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda command, **kwargs: original if command[-1].startswith("a" * 40) else changed,
    )
    with pytest.raises(ValueError, match="experiment conditions"):
        runner._operation_code(run, config, revision)


def test_before_spawn_ebadf_records_failure_and_preserves_data(resumable, monkeypatch):
    _, run, _, _, _ = resumable
    runner._approve_operations(run)
    before = digest(run / "data/games/000022.json.gz")
    calls = []

    def fail(*args, **kwargs):
        calls.append(args)
        raise OSError(errno.EBADF, "injected before spawning supervisor")

    monkeypatch.setattr(runner, "_launch", fail)
    with pytest.raises(OSError, match="injected"):
        runner.start(run, recover_startup=True)
    state = runner._state(run)
    assert state["status"] == "needs_astra"
    assert state["startup_failure"] == "before_spawn"
    assert "injected before spawning supervisor" in state["errors"][0]["traceback"]
    assert digest(run / "data/games/000022.json.gz") == before
    assert len(calls) == 1
    with runner._lease(run):
        pass


def test_interrupted_pending_start_can_resume_without_changing_ready_by_hand(resumable):
    _, run, _, _, _ = resumable
    runner._approve_operations(run)
    with runner._lease(run):
        _, pending = runner._prepare_resume(run)
    # Simulate the CLI disappearing after durable preparation and before spawn:
    # no child or lease exists, and the on-disk state is still startup_pending.
    assert pending["status"] == "running"
    assert pending["reason"] == "startup_pending"
    assert "pid" not in pending
    with runner._lease(run):
        _, resumed = runner._prepare_resume(run)
    assert resumed["execution_attempt"] == 2
    assert resumed["operation_revision"] == pending["operation_revision"]
    assert resumed["began_at"] == pending["began_at"]
    record = json.loads((run / "attempts/000002.json").read_text())
    assert record["previous_state"] == pending


def test_attempt_result_write_failure_never_publishes_ready(resumable, monkeypatch):
    _, run, _, _, _ = resumable
    runner._approve_operations(run)
    with runner._lease(run):
        runner._prepare_resume(run)
    atomic(
        run / "generation-probe.json",
        encoded({"run_sha256": digest(run / "run.json"), "result": {"games": 1}}),
    )
    original_atomic = runner.atomic

    def fail_result(path, value):
        if path.name.endswith("-result.json"):
            raise OSError(errno.EBADF, "injected result persistence failure")
        return original_atomic(path, value)

    monkeypatch.setattr(runner, "_run_stage", lambda *args: (0, None))
    monkeypatch.setattr(runner, "atomic", fail_result)
    result = runner.work(run, pause_after_new_games=1)
    assert result["status"] == "needs_astra"
    assert result["reason"] == "attempt_result_write_failed"
    assert "injected result persistence failure" in result["errors"][-1]["traceback"]
    assert runner._state(run)["status"] == "needs_astra"
    with runner._lease(run), pytest.raises(ValueError, match="outside startup recovery"):
        runner._prepare_resume(run)


@pytest.mark.parametrize("reason", ["data_write_ebadf", "teacher_quality_failure"])
def test_resume_and_pause_cannot_clear_unknown_failure(resumable, reason):
    _, run, _, _, _ = resumable
    runner._approve_operations(run)
    state = runner._state(run)
    state.update(status="needs_astra", reason=reason)
    atomic(run / "state.json", encoded(state))
    before = (run / "state.json").read_bytes()
    with runner._lease(run), pytest.raises(ValueError, match="outside startup recovery"):
        runner._prepare_resume(run)
    with pytest.raises(ValueError, match="cannot clear"):
        runner.pause(run)
    assert (run / "state.json").read_bytes() == before


def test_resume_while_lease_is_held_never_starts_a_second_writer(resumable, monkeypatch):
    _, run, _, _, _ = resumable
    runner._approve_operations(run)

    def forbidden(*args, **kwargs):
        pytest.fail("second launcher must not run while the lease is held")

    monkeypatch.setattr(runner, "_launch", forbidden)
    with runner._lease(run):
        result = runner.start(run, recover_startup=True)
    assert result["lease_held"] is True
    assert not list((run / "attempts").glob("*.json"))
