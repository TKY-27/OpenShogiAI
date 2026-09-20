"""Distinct updated candidates survive best0, optional gaps, and exact resume."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from open_shogi_training import evaluator_training as training
from open_shogi_training.evaluator_sampling import coverage_order


def paired_dataset(tmp_path):
    from test_evaluator_training import corpus, make_dataset, setup
    from test_r4_c2 import policy

    data, config = setup(tmp_path)
    for split in ("train", "validation"):
        make_dataset(data, corpus()[:8] * 4, split)
        np.save(data / f"{split}-groups.npy", np.arange(32) // 2 % 4)
        np.save(data / f"{split}-sources.npy", np.array([0] * 16 + [1] * 16))
        np.save(data / f"{split}-origins.npy", np.array([0] * 16 + [2] * 16))
        np.save(data / f"{split}-sequences.npy", np.arange(32) // 2)
        np.save(data / f"{split}-partners.npy", np.arange(32, dtype=np.int32) ^ 1)
    config.update(
        policy(),
        max_steps=4,
        max_epochs=8,
        validation_every=2,
        patience=10,
        minimum_relative_improvement=0.002,
        maximum_replay_regression_ratio=1.03,
        source_fractions=[0.5, 0.5],
        pair_weight=0.25,
        trained_candidate="best_updated_objective_v1",
    )
    config["coverage_sampler"].update(epoch_examples=32, maximum_per_example=[8, 8])
    return data, config


def test_pair_direction_ties_missing_and_bounded_sampler(tmp_path):
    target = torch.tensor([[-300.0, 300.0], [30.0, 0.0], [6000.0, 0.0]])
    correct = torch.tensor([[-300.0, 300.0], [-1000.0, 1000.0], [700.0, 0.0]], requires_grad=True)
    assert training.pair_loss(correct, target).sum() == 0
    wrong = -correct
    loss = training.pair_loss(wrong, target)
    assert loss[0] > 0 and loss[1] == 0 and loss[2] > 0
    loss.sum().backward()
    assert torch.isfinite(correct.grad).all()
    data, config = paired_dataset(tmp_path)
    arrays = training.grouped_arrays(data, "train")
    counts = torch.zeros(32, dtype=torch.int32)
    order = coverage_order(arrays, counts, config, torch.Generator().manual_seed(1))
    assert len(order) == len(order.unique()) == 32
    assert sum(len(training.batch_pairs(arrays, b.numpy())) for b in order.split(16)) == 16
    assert training.batch_pairs(arrays, np.array([0, 3])).shape == (0, 2)
    partners = np.load(data / "train-partners.npy")
    partners[0] = 3
    np.save(data / "train-partners.npy", partners)
    with pytest.raises(ValueError, match="reciprocal"):
        training.grouped_arrays(data, "train")


def test_best0_keeps_distinct_updated_candidate_and_exact_resume(tmp_path, monkeypatch):
    data, config = paired_dataset(tmp_path)
    original = training.evaluate_groups
    from open_shogi_training.phase10v_model import Phase10VModel, torch_parameters

    baseline = torch_parameters(Phase10VModel.read(Path(config["initial_model"])))[0]

    def worsens(parameters, *args, **kwargs):
        result = original(parameters, *args, **kwargs)
        value = 1.0 if torch.equal(parameters[0], baseline) else 2.0
        for metric in result.values():
            metric["loss"] = value
        return result

    monkeypatch.setattr(training, "evaluate_groups", worsens)
    # Force the scalar incumbent to remain step0; no change to actual gradients/export.
    identity = {"fixture": "c3"}
    paused = training.train(data, tmp_path / "fit", config, identity, stop_after=1)
    assert paused["step"] == 1
    result = training.train(data, tmp_path / "fit", config, identity)
    assert result["best_step"] == 0
    assert result["trained_candidate"]["step"] > 0
    assert result["trained_candidate"]["sha256"] != result["incumbent"]["sha256"]
    ref = json.loads((tmp_path / "fit/resume.json").read_text())
    state = torch.load(tmp_path / "fit" / ref["path"], weights_only=True)
    assert int(state["counts"].sum()) == result["example_exposures"]
    assert (tmp_path / "fit/trained_candidate.osaval03").read_bytes() == state[
        "trained_model_bytes"
    ]
    assert all(int(s["step"]) == 4 for s in state["optimizer"]["state"].values())
    assert training.train(data, tmp_path / "fit", config, identity) == result

    whole = training.train(data, tmp_path / "whole", config, identity)
    assert whole["trained_candidate"] == result["trained_candidate"]
    full_ref = json.loads((tmp_path / "whole/resume.json").read_text())
    full = torch.load(tmp_path / "whole" / full_ref["path"], weights_only=True)
    for key in ("counts", "order", "sampler_rng", "torch_rng"):
        assert torch.equal(state[key], full[key])
    for actual, expected in zip(state["parameters"], full["parameters"], strict=True):
        assert torch.equal(actual, expected)


def test_updated_route_registers_when_incumbent_best0_and_optional_screen_missing(
    tmp_path, monkeypatch
):
    from open_shogi_training import evaluator_development as development
    from open_shogi_training import evaluator_run as runner
    from open_shogi_training.evaluator_data import atomic, digest, encoded

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    run = tmp_path / "local/run"
    atomic(run / "fit/best.osaval03", b"incumbent-step0")
    candidate = run / "fit/trained_candidate.osaval03"
    atomic(candidate, b"updated-tensors")
    config = {
        "run_id": "c3-fixture",
        "round": {"id": "R4-C3"},
        "training": {"trained_candidate": "best_updated_objective_v1"},
        "development_integration": {
            "selection": "r4c3",
            "port": 5175,
            "ui_identity": {"files": {}},
        },
        "runtime": {},
    }
    for k, p in runner.RUNTIME_SOURCES.items():
        atomic(tmp_path / p, b"runtime")
        config["runtime"][k] = {"path": p, "sha256": digest(tmp_path / p)}
    atomic(run / "run.json", encoded(config))
    atomic(run / "arena/arena.json", encoded({"adoption_criteria_met": False}))
    atomic(
        run / "development-test.json",
        encoded(
            {
                "groups": {
                    name: {g: {"loss": loss} for g in training.GROUPS}
                    for name, loss in (("baseline", 1.0), ("candidate", 1.1))
                },
                "move_quality_screen": {"screen_pass": None},
            }
        ),
    )
    review = runner._candidate_review(run)
    assert review["candidate_sha256"] == digest(candidate)
    assert not review["meets_frozen_criteria"]
    assert review["move_quality_screen_pass"] is None
    atomic(run / "candidate-review.json", encoded(review))
    output = run / "development"

    def command(args, **kwargs):
        if "check_evaluator_model.mjs" in args[1]:
            assert Path(args[3]) == candidate
            atomic(output / "model-audit.json", encoded({"status": "PASS"}))
        else:
            assert args[3] == "r4c3"
            atomic(
                output / "browser/browser.json",
                encoded({"status": "PASS", "expectedHash": digest(candidate)}),
            )

    from contextlib import nullcontext
    from types import SimpleNamespace

    monkeypatch.setattr(development.subprocess, "run", command)
    monkeypatch.setattr(
        development.urllib.request,
        "urlopen",
        lambda *a, **k: nullcontext(SimpleNamespace(status=200)),
    )
    registered = development.register(tmp_path, run, config, runner.selected_model(run), output)
    assert registered["status"] == "PASS"
    assert registered["model"]["sha256"] == digest(candidate)
    descriptor = json.loads((tmp_path / "local/core-prototype/r4c3.json").read_text())
    assert descriptor["leaf"]["sha256"] == digest(candidate)
