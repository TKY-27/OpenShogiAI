"""Allocation fallback retains one effective batch and rejects unrelated failures."""

import json

import numpy as np
import pytest
import torch
from open_shogi_training import evaluator_training as training


def fixture(monkeypatch):
    data = {
        "features": np.arange(14, dtype=np.int64).reshape(7, 2, 1),
        "lengths": np.ones((7, 2), dtype=np.int64),
        "targets": np.arange(7, dtype=np.float32) * 90,
    }
    monkeypatch.setattr(training, "forward", lambda p, x, _: x[:, :, 0].float() @ p[0] + p[1])
    return data, [
        torch.nn.Parameter(torch.tensor([2.0, -3.0])),
        torch.nn.Parameter(torch.tensor(1.0)),
    ]


def test_accumulation_matches_effective_batch_and_discards_failed_partial_gradients(monkeypatch):
    data, parameters = fixture(monkeypatch)
    indexes = np.array([5, 0, 3, 1, 2, 6, 4])
    total, _ = training.accumulate(parameters, data, indexes, 7)
    expected = [p.grad.clone() for p in parameters]
    original = training.forward
    calls = []

    def fail_after_partial(p, x, lengths):
        calls.append(len(x))
        if len(calls) == 2:
            raise MemoryError("injected actual allocation failure")
        return original(p, x, lengths)

    monkeypatch.setattr(training, "forward", fail_after_partial)
    actual, size = training.accumulate(parameters, data, indexes, 3)
    assert size == 1 and calls[:2] == [3, 3]
    assert actual == pytest.approx(total, rel=1e-6)
    for p, gradient in zip(parameters, expected, strict=True):
        torch.testing.assert_close(p.grad, gradient)
    reference_parameters = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    for p, gradient in zip(reference_parameters, expected, strict=True):
        p.grad = gradient
    reference_optimizer = torch.optim.AdamW(reference_parameters, lr=0.001)
    reference_optimizer.step()
    optimizer = torch.optim.AdamW(parameters, lr=0.001)
    optimizer.step()
    assert all(state["step"] == 1 for state in optimizer.state.values())
    for p, reference in zip(parameters, reference_parameters, strict=True):
        torch.testing.assert_close(p, reference)


def test_exhaustion_is_finite_and_other_errors_are_not_memory(monkeypatch):
    data, parameters = fixture(monkeypatch)
    calls = []

    def fail(_p, x, _lengths):
        calls.append(len(x))
        raise torch.OutOfMemoryError("injected")

    monkeypatch.setattr(training, "forward", fail)
    with pytest.raises(training.TrainingResourceWaitError, match="microbatch 1"):
        training.accumulate(parameters, data, np.arange(7), 7)
    assert calls == [7, 3, 1]
    assert all(p.grad is None for p in parameters)
    assert not training.allocation_failure(RuntimeError("exit 1 SIGKILL"))
    assert not training.allocation_failure(FloatingPointError("nonfinite"))
    monkeypatch.setattr(training, "forward", lambda *_: (_ for _ in ()).throw(RuntimeError("bug")))
    with pytest.raises(RuntimeError, match="bug"):
        training.accumulate(parameters, data, np.arange(7), 7)


def test_validation_fallback_and_group_selection_do_not_copy_entire_group(monkeypatch):
    data, parameters = fixture(monkeypatch)
    indexes = np.array([1, 3, 6])
    expected = training.evaluate(parameters, data, 3, indexes=indexes)
    original = training.forward

    def limited(p, x, lengths):
        if len(x) > 1:
            raise MemoryError("injected")
        return original(p, x, lengths)

    monkeypatch.setattr(training, "forward", limited)
    actual = training.evaluate(parameters, data, 3, indexes=indexes)
    assert actual == pytest.approx(expected, rel=1e-6)


def test_fallback_checkpoint_restores_sampler_optimizer_and_microbatch(tmp_path, monkeypatch):
    from test_evaluator_training import setup

    data, config = setup(tmp_path)
    config.update(max_steps=3, microbatch_size=3)
    original = training.forward

    def limited(p, x, lengths):
        if torch.is_grad_enabled() and len(x) > 1:
            raise MemoryError("injected")
        return original(p, x, lengths)

    monkeypatch.setattr(training, "forward", limited)
    whole = tmp_path / "whole"
    resumed = tmp_path / "resumed"
    training.train(data, whole, config, {"fixture": True})
    training.train(data, resumed, config, {"fixture": True}, stop_after=1)
    monkeypatch.setattr(training, "forward", original)
    training.train(data, resumed, config, {"fixture": True})
    expected = torch.load(
        whole / json.loads((whole / "resume.json").read_text())["path"], weights_only=True
    )
    actual = torch.load(
        resumed / json.loads((resumed / "resume.json").read_text())["path"], weights_only=True
    )
    assert actual["microbatch_size"] == expected["microbatch_size"] == 1
    assert actual["step"] == 3 and actual["exposures"] == expected["exposures"] == 7
    assert actual["offset"] == expected["offset"]
    for key in ("counts", "order", "sampler_rng", "torch_rng"):
        assert torch.equal(actual[key], expected[key])
    for value, reference in zip(actual["parameters"], expected["parameters"], strict=True):
        assert torch.equal(value, reference)
    for key, state in actual["optimizer"]["state"].items():
        for name, value in state.items():
            assert torch.equal(value, expected["optimizer"]["state"][key][name])


def test_invalid_targets_are_rejected_before_forward(monkeypatch):
    data, parameters = fixture(monkeypatch)
    data["targets"][0] = np.nan
    monkeypatch.setattr(training, "forward", lambda *_: pytest.fail("invalid target reached loss"))
    with pytest.raises(ValueError, match="nonfinite"):
        training.accumulate(parameters, data, np.array([0]), 1)
    assert all(p.grad is None for p in parameters)
    with pytest.raises(ValueError, match="empty"):
        training.accumulate(parameters, data, np.array([], dtype=int), 1)


def test_each_update_is_durable_before_next_optimizer_and_resume(tmp_path, monkeypatch):
    from test_evaluator_training import setup

    data, config = setup(tmp_path)
    config.update(max_steps=3, validation_every=3)
    folder = tmp_path / "fit"
    original = torch.optim.AdamW.step
    calls = 0

    def interrupted(optimizer, *args, **kwargs):
        nonlocal calls
        calls += 1
        reference = json.loads((folder / "resume.json").read_text())
        assert reference["step"] == calls - 1
        if calls == 2:
            with torch.no_grad():
                optimizer.param_groups[0]["params"][0].add_(1)
            raise torch.OutOfMemoryError("injected partially mutated AdamW")
        return original(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", interrupted)
    with pytest.raises(training.TrainingResourceWaitError, match="optimizer"):
        training.train(data, folder, config, {"fixture": True})
    reference = json.loads((folder / "resume.json").read_text())
    assert reference["step"] == 1
    checkpoint = torch.load(folder / reference["path"], weights_only=True)
    assert checkpoint["exposures"] == 3
    monkeypatch.setattr(torch.optim.AdamW, "step", original)
    actual = training.train(data, folder, config, {"fixture": True})
    expected = training.train(data, tmp_path / "whole", config, {"fixture": True})
    assert actual["best_sha256"] == expected["best_sha256"]
    assert actual["example_exposures"] == expected["example_exposures"] == 7


def test_checkpoint_allocation_failure_cannot_automatically_replay_success(tmp_path, monkeypatch):
    from test_evaluator_training import setup

    data, config = setup(tmp_path)
    original = training._save_checkpoint

    def interrupted(folder, state):
        if state["step"] == 1:
            raise MemoryError("injected serialization allocation")
        return original(folder, state)

    monkeypatch.setattr(training, "_save_checkpoint", interrupted)
    with pytest.raises(training.CheckpointPublicationError, match="after step 1") as captured:
        training.train(data, tmp_path / "fit", config, {"fixture": True})
    assert not training.allocation_failure(captured.value)


def test_same_step_publication_failure_preserves_old_checkpoint_reference(tmp_path, monkeypatch):
    original = training.atomic
    training._save_checkpoint(tmp_path, {"step": 1, "value": torch.tensor(1)})
    reference = json.loads((tmp_path / "resume.json").read_text())

    def interrupted(path, payload):
        if path.name == "resume.json":
            raise OSError("injected publication failure")
        original(path, payload)

    monkeypatch.setattr(training, "atomic", interrupted)
    with pytest.raises(OSError, match="publication"):
        training._save_checkpoint(tmp_path, {"step": 1, "value": torch.tensor(2)})
    assert json.loads((tmp_path / "resume.json").read_text()) == reference
    assert training.digest(tmp_path / reference["path"]) == reference["sha256"]
    monkeypatch.setattr(training, "atomic", original)
    training._save_checkpoint(tmp_path, {"step": 1, "value": torch.tensor(3)})
    current = json.loads((tmp_path / "resume.json").read_text())
    assert training.digest(tmp_path / current["path"]) == current["sha256"]
    assert len(list(tmp_path.glob("checkpoint-*.pt"))) == 2


def test_resume_completes_pending_validation_before_next_update(tmp_path, monkeypatch):
    from test_evaluator_training import setup

    data, config = setup(tmp_path)
    config.update(max_steps=2, validation_every=1)
    folder = tmp_path / "fit"
    original = training.evaluate
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise training.TrainingResourceWaitError("injected validation allocation")
        return original(*args, **kwargs)

    monkeypatch.setattr(training, "evaluate", interrupted)
    with pytest.raises(training.TrainingResourceWaitError):
        training.train(data, folder, config, {"fixture": True})
    assert json.loads((folder / "resume.json").read_text())["step"] == 1
    monkeypatch.setattr(training, "evaluate", original)
    actual = training.train(data, folder, config, {"fixture": True})
    expected = training.train(data, tmp_path / "whole", config, {"fixture": True})
    assert [event["step"] for event in actual["history"]] == [0, 1, 2]
    assert actual["best_sha256"] == expected["best_sha256"]
    assert actual["example_exposures"] == expected["example_exposures"]
