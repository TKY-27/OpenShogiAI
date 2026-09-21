"""Finite C4 orchestration, local missing data, and exact updated-prefix recovery."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from open_shogi_training import evaluator_training as training
from open_shogi_training import r4_c4 as c4
from open_shogi_training import r4_c4_data as data
from open_shogi_training.evaluator_data import atomic, encoded
from open_shogi_training.evaluator_ledger import Ledger
from open_shogi_training.labeling.usi import USIIncompleteDepthError, USIProcessError


def test_teacher_missing_exit_and_interruption_are_finite(tmp_path, monkeypatch):
    config = SimpleNamespace(binary_sha256="a" * 64)
    teacher = data.Teacher.__new__(data.Teacher)
    teacher.root, teacher.config = tmp_path, config
    teacher.policy = dict(
        broad_depth=8, strong_depth=12, broad_nodes=200, strong_nodes=2000, maximum_worker_exits=3
    )
    teacher.ledger, teacher.engine = Ledger(tmp_path), None
    failure = USIIncompleteDepthError
    starts = []

    class Engine:
        def __init__(self, *args):
            pass

        def start(self):
            starts.append(1)

        def close(self):
            pass

        def analyze(self, *args, **kwargs):
            assert kwargs["auto_start"] is False
            raise failure("injected local missing label")

    monkeypatch.setattr(data, "USIEngine", Engine)
    key, value = teacher.observe("fixture", {"game": 0})
    assert value is None
    assert teacher.ledger.get(key)["status"] == "deferred"
    teacher.observe("fixture", {"game": 0})
    assert len(teacher.ledger.get(key)["attempts"]) == 1
    failure = USIProcessError
    for game in range(1, 4):
        assert teacher.observe("fixture", {"game": game})[1] is None
    with pytest.raises(RuntimeError, match="systemic"):
        teacher.observe("fixture", {"game": 4})
    assert teacher.ledger.db.execute("SELECT value FROM counters").fetchone()[0] == 4
    teacher.ledger.close()
    teacher.ledger = Ledger(tmp_path)
    assert teacher.observe("fixture", {"game": 4})[1]["status"] == "missing"
    assert teacher.ledger.db.execute("SELECT value FROM counters").fetchone()[0] == 4
    teacher.ledger.close()


def test_c4_origin_loss_and_missing_pair_validation_resume_exactly(tmp_path):
    from test_r4_c3 import paired_dataset

    dataset, config = paired_dataset(tmp_path)
    config.update(track_origin_loss=True, allow_missing_validation_signals=True)
    config["coverage_sampler"]["origin_fractions"] = {"0": 0.5, "4": 0.25, "5": 0.25}
    for split in ("train", "validation"):
        np.save(dataset / f"{split}-origins.npy", np.array([2] * 16 + [4] * 8 + [5] * 8))
    np.save(dataset / "validation-partners.npy", np.full(32, -1, dtype=np.int32))
    identity = {"fixture": "c4"}
    assert training.train(dataset, tmp_path / "fit", config, identity, stop_after=1)["step"] == 1
    result = training.train(dataset, tmp_path / "fit", config, identity)
    whole = training.train(dataset, tmp_path / "whole", config, identity)
    assert result["trained_candidate"] == whole["trained_candidate"]
    saved = []
    for folder in ("fit", "whole"):
        ref = data.read(tmp_path / folder / "resume.json")
        saved.append(torch.load(tmp_path / folder / ref["path"], weights_only=True))
    assert saved[0]["loss_contributions"] == saved[1]["loss_contributions"]
    assert set(saved[0]["loss_contributions"]["scalar"]) == {"2", "4", "5"}
    assert (
        sum(v["examples"] for v in saved[0]["loss_contributions"]["scalar"].values())
        == result["example_exposures"]
    )
    c4.changed(Path(config["initial_model"]), tmp_path / "fit/trained_candidate.osaval03")
    with pytest.raises(ValueError, match="no quantized"):
        c4.changed(Path(config["initial_model"]), Path(config["initial_model"]))


@pytest.mark.parametrize("admit,generations", [(False, 3), (True, 5)])
def test_outer_generations_continue_after_nonadoption_and_optional_missing(
    tmp_path, monkeypatch, admit, generations
):
    from open_shogi_training import evaluator_development

    config = json.loads((Path(__file__).parents[2] / "configs/evaluator-main.json").read_text())
    config["code"] = {"commit": "fixture-not-strength-evidence"}
    run = tmp_path / "local/run"
    config["output"] = "local/run"
    initial = tmp_path / "initial.osaval03"
    atomic(initial, b"fixture-initial")
    initial_ref = data.reference(tmp_path, initial)
    config["generation"].update(leaf_path=initial_ref["path"], leaf_sha256=initial_ref["sha256"])
    config["iteration"]["final_baselines"] = {"c3": initial_ref, "defense": initial_ref}
    atomic(run / "run.json", encoded(config))
    produced = []
    fits = []

    def generate(root, parent, folder, cfg, actor, spec):
        value = {"status": "complete", "artifacts": []}
        if not (folder / "result.json").exists():
            produced.append((spec["id"], copy.deepcopy(actor), spec["index"]))
        atomic(folder / "result.json", encoded(value))
        return value

    def build(root, parent, folder, cfg, spec, generated):
        value = {"new_unique_positions": {"train": 100001}}
        atomic(folder / "dataset/manifest.json", encoded(value))
        return value

    def train(dataset, fit, cfg, identity, **kwargs):
        fits.append(copy.deepcopy(cfg))
        value = {
            "status": "complete",
            "example_exposures": 1024,
            "seen_unique_positions": 800,
            "reason": "validation_patience",
        }
        atomic(fit / "training.json", encoded(value))
        atomic(fit / "trained_candidate.osaval03", str(fit).encode())
        return value

    def audit(root, parent, cfg, model, folder):
        atomic(folder / "model-audit.json", encoded({"status": "PASS"}))

    def compare(root, parent, cfg, folder, candidate, baseline, dataset, confirmation=None):
        value = {
            "status": "complete",
            "attempts": [],
            "summary": {
                "all_planned_complete": True,
                "adverse_attempts": 0,
                "paired_bootstrap_one_sided_95_lower": 0.6 if admit else 0.3,
                "candidate_score_by_clock": {
                    "180000": 0.7 if admit else 0.4,
                    "600000": 0.6 if admit else 0.4,
                },
            },
        }
        atomic(folder / "arena.json", encoded(value))
        return value

    registered = []

    def register(root, parent, cfg, candidate, folder):
        registered.append(candidate)
        return {"status": "PASS"}

    monkeypatch.setattr(c4, "generate", generate)
    monkeypatch.setattr(c4, "build", build)
    monkeypatch.setattr(c4, "train", train)
    monkeypatch.setattr(c4, "changed", lambda *a: [1])
    monkeypatch.setattr(c4, "audit", audit)
    monkeypatch.setattr(c4, "grouped_arrays", lambda *a: None)
    monkeypatch.setattr(c4, "torch_parameters", lambda *a: None)
    monkeypatch.setattr(c4.Phase10VModel, "read", lambda *a: None)
    monkeypatch.setattr(c4, "evaluate_groups", lambda *a: {g: {"loss": 1} for g in training.GROUPS})
    monkeypatch.setattr(c4, "compare", compare)
    monkeypatch.setattr(c4, "confirmation_starts", lambda *a: None)
    monkeypatch.setattr(evaluator_development, "register", register)
    result = c4.execute(tmp_path, run, config)
    assert len(result["decisions"]) == generations
    assert result["optional_screen_pass"] is None
    assert result["novel_confirmation"] == "missing_unverified"
    assert not result["human_shodan_validated"] and not result["promotion_performed"]
    assert len(registered) == 1 and result["candidate"] != initial_ref
    assert len(produced) == generations + 1
    assert len({seed for _, _, seed in produced}) == generations + 1
    if not admit:
        assert all(actor == initial_ref for _, actor, _ in produced)
        assert fits[-1]["learning_rate"] == config["training"]["learning_rate"] / 2
    else:
        assert produced[2][1] != initial_ref
    c4.verify_results(tmp_path, run)
    assert c4.execute(tmp_path, run, config) == result
    assert len(produced) == generations + 1
    (tmp_path / result["candidate"]["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="changed"):
        c4.verify_results(tmp_path, run)


def test_c4_terminal_state_uses_iteration_receipt_and_nested_stop(tmp_path, monkeypatch):
    from open_shogi_training import evaluator_arena, evaluator_run

    monkeypatch.setattr(evaluator_run, "ROOT", tmp_path)
    config = {"run_id": "c4-test", "iteration": {"schema": "fixture"}, "generation": {}}
    atomic(tmp_path / "run.json", encoded(config))
    atomic(tmp_path / "seal.json", encoded({"run_id": "c4-test"}))
    atomic(tmp_path / "c4-result.json", encoded({"status": "complete"}))
    from open_shogi_training.evaluator_data import digest

    state = {
        "run_id": "c4-test",
        "status": "awaiting_astra_review",
        "result_sha256": digest(tmp_path / "c4-result.json"),
    }
    atomic(tmp_path / "state.json", encoded(state))
    seen = []
    monkeypatch.setattr(evaluator_run, "_verify_completion", lambda r, s, c: seen.append(s))
    assert evaluator_run._state(tmp_path) == state
    assert seen == ["iterate"] and evaluator_run._stages(config) == ("iterate",)
    nested = tmp_path / "generations/G01/arena"
    nested.mkdir(parents=True)
    assert not evaluator_arena._stopping(nested)
    atomic(tmp_path / "STOP", b"user pause")
    assert evaluator_arena._stopping(nested)


def test_strong_analysis_reserves_contact_and_late_game_budget():
    assert [data.strong_budget(p, 8) for p in (10, 48, 120)] == [(0, 2), (1, 4), (2, 2)]
    assert [data.strong_budget(p, 2)[1] for p in (10, 48, 120)] == [1, 1, 0]
    assert all(data.strong_budget(p, 0)[1] == 0 for p in (10, 48, 120))
