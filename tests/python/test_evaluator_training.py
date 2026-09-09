"""Exact resume across shuffled epochs, typed child labels and protected split contracts."""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from open_shogi_training.evaluator_data import (
    START,
    _observations,
    atomic,
    digest,
    encoded,
    partition,
    prepare,
    symmetry_keys,
)
from open_shogi_training.evaluator_training import forward, train
from open_shogi_training.phase10u_execution import successor_sfen
from open_shogi_training.phase10v_model import (
    Phase10VModel,
    sparse_features,
    torch_forward,
    torch_parameters,
)


def corpus():
    lines = [START]
    sfen = START
    for move in ["7g7f", "3c3d", "5i6h", "5a4b", "2g2f", "8c8d", "8h2b+", "3a2b"]:
        sfen = successor_sfen(sfen, move)
        lines.append(sfen)
    return lines


def make_dataset(folder: Path, sfens: list[str], split: str):
    ids = np.zeros((len(sfens), 2, 48), dtype=np.uint16)
    lengths = np.zeros((len(sfens), 2), dtype=np.uint8)
    for i, sfen in enumerate(sfens):
        black, white, stm = sparse_features(sfen)
        for side, features in enumerate((white, black) if stm else (black, white)):
            ids[i, side, : len(features)] = features
            lengths[i, side] = len(features)
    np.save(folder / f"{split}-features.npy", ids)
    np.save(folder / f"{split}-lengths.npy", lengths)
    np.save(folder / f"{split}-targets.npy", np.arange(len(sfens), dtype=np.float32) * 150 - 200)
    return ids, lengths


def setup(tmp_path):
    model = Phase10VModel.random(seed=17)
    path = tmp_path / "initial.osaval03"
    model.write(path)
    data = tmp_path / "data"
    data.mkdir()
    make_dataset(data, corpus()[:7], "train")
    make_dataset(data, corpus()[7:], "validation")
    config = {
        "device": "cpu",
        "threads": 1,
        "seed": 23,
        "initial_model": str(path),
        "initial_sha256": model.sha256,
        "batch_size": 3,
        "max_steps": 10,
        "max_epochs": 5,
        "validation_every": 3,
        "patience": 20,
        "minimum_relative_improvement": 0.0,
        "learning_rate": 0.0002,
        "minimum_learning_rate": 0.00001,
        "warmup_steps": 2,
        "gradient_norm_limit": 5.0,
    }
    return data, config


def test_preencoded_q20_matches_reference_and_all_scalar_layers_have_gradients(tmp_path):
    model = Phase10VModel.random(seed=17)
    p = torch_parameters(model)
    ids, lengths = make_dataset(tmp_path, corpus(), "train")
    result = forward(p, torch.tensor(ids.astype(np.int64)), torch.tensor(lengths.astype(np.int64)))
    reference = torch_forward(p, corpus())[:, 0]
    torch.testing.assert_close(result, reference, atol=0, rtol=0)
    result.square().mean().backward()
    for parameter in p:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_resume_mid_epoch_is_bit_exact_and_counts_unique_exposures(tmp_path):
    data, config = setup(tmp_path)
    identity = {"dataset": "fixture", "code": "fixture"}
    complete = train(data, tmp_path / "whole", config, identity)
    paused = train(data, tmp_path / "resumed", config, identity, stop_after=4)
    assert paused["step"] == 4
    resumed = train(data, tmp_path / "resumed", config, identity)
    assert resumed["best_sha256"] == complete["best_sha256"]
    for folder in ["whole", "resumed"]:
        ref = json.loads((tmp_path / folder / "resume.json").read_text())
        state = torch.load(tmp_path / folder / ref["path"], weights_only=True)
        if folder == "whole":
            expected = state
        else:
            for key in ("order", "counts", "sampler_rng", "torch_rng"):
                assert torch.equal(state[key], expected[key])
            for a, b in zip(state["parameters"], expected["parameters"], strict=True):
                assert torch.equal(a, b)
            assert state["offset"] == expected["offset"]
    assert resumed["seen_unique_positions"] == 7
    assert resumed["example_exposures"] == 24
    assert resumed["maximum_exposure"] == 4
    assert train(data, tmp_path / "resumed", config, identity) == resumed
    with pytest.raises(ValueError, match="identity changed"):
        train(data, tmp_path / "resumed", dict(config, seed=24), identity)


def test_checkpoint_binds_best_and_restores_only_bound_export(tmp_path):
    data, config = setup(tmp_path)
    folder = tmp_path / "fit"
    train(data, folder, config, {"code": "fixture"}, stop_after=2)
    (folder / "best.osaval03").write_bytes(b"corrupted")
    result = train(data, folder, config, {"code": "fixture"})
    assert (
        hashlib.sha256((folder / "best.osaval03").read_bytes()).hexdigest() == result["best_sha256"]
    )
    assert next(iter(folder.glob("uncommitted-best-*.osaval03"))).read_bytes() == b"corrupted"
    ref = json.loads((folder / "resume.json").read_text())
    (folder / ref["path"]).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checkpoint hash"):
        train(data, folder, config, {"code": "fixture"})


def test_child_perspective_and_terminal_masks():
    sfen = successor_sfen(START, "7g7f")
    candidate = {
        "pv": ["7g7f"],
        "child_sfen": sfen,
        "score": {"kind": "cp", "value": 123},
        "child_terminal": "None",
    }
    game = {"records": [{"sfen": START, "ply": 0, "candidates": [candidate], "deviation": None}]}
    rows = list(_observations(game))
    assert rows[0][1]["value"] == 123
    assert rows[1][1]["value"] == -123
    candidate["child_terminal"] = "Some(NoLegalMoves)"
    assert len(list(_observations(game))) == 1


def test_game_split_precedes_features_and_symmetry_dedup():
    assert {partition(i, 20260915) for i in range(8)} == {"train", "validation", "development_test"}
    assert symmetry_keys(START) == symmetry_keys(START[:-1] + "87")
    rotated = START.replace(" b ", " w ")
    assert symmetry_keys(START) == symmetry_keys(rotated)


def test_prepare_excludes_cross_game_symmetries_and_rechecks_artifact_bytes(tmp_path):
    output = tmp_path / "data"
    guard = tmp_path / "guard.json"
    atomic(guard, encoded({}))
    development = tmp_path / "known.json"
    atomic(development, encoded({"symmetry_keys": sorted(symmetry_keys(corpus()[1]))}))
    config = {
        "seed": 20260915,
        "games": 8,
        "split_guard_path": guard.name,
        "split_guard_sha256": digest(guard),
        "development_exclusions_path": development.name,
        "development_exclusions_sha256": digest(development),
    }
    atomic(output / "generation.json", encoded(config))
    atomic(output / "generation-complete.json", encoded({"games": 8}))
    for game in range(8):
        records = []
        for sfen in (START, corpus()[game + 1]):
            records.append(
                {
                    "sfen": sfen,
                    "ply": game,
                    "deviation": None,
                    "candidates": [
                        {"score": {"kind": "cp", "value": 100}, "child_terminal": "Some(Checkmate)"}
                    ],
                }
            )
        raw = {
            "game": game,
            "seed": config["seed"] + game,
            "split": partition(game, config["seed"]),
            "records": records,
        }
        path = output / "games" / f"{game:06d}.json.gz"
        atomic(path, gzip.compress(encoded(raw)))
        atomic(path.with_suffix(".receipt.json"), encoded({"sha256": digest(path)}))
    report = prepare(tmp_path, output, config, [])
    assert report["excluded_conflict_keys"] >= 2
    seen = set()
    for path in (output / "dataset").glob("*-rows.jsonl.gz"):
        with gzip.open(path, "rt") as stream:
            keys = {json.loads(line)["symmetry_key"] for line in stream}
        assert not seen & keys
        seen.update(keys)
    assert min(symmetry_keys(START)) not in seen
    assert min(symmetry_keys(corpus()[1])) not in seen
    assert prepare(tmp_path, output, config, []) == report
    target = output / "dataset" / "train-targets.npy"
    original = target.read_bytes()
    target.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(ValueError, match="dataset corrupted"):
        prepare(tmp_path, output, config, [])
    target.write_bytes(original)
    (output / "games" / "000000.json.gz").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="source trajectory corrupted"):
        prepare(tmp_path, output, config, [])
