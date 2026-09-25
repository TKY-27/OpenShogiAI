"""C2 sampling/resume, small improvements, and independently decoded public bytes."""

import json
import urllib.request
from collections import Counter

import numpy as np
import pytest
import torch
from open_shogi_training import evaluator_training as training
from open_shogi_training import r4_sources
from open_shogi_training.evaluator_sampling import coverage_order, exposure_summary
from open_shogi_training.r4_sources import packed_move, unpack_position


def policy():
    return {
        "batch_size": 16,
        "sampling_fractions": [0.25] * 4,
        "source_fractions": [0.25, 0.75],
        "coverage_sampler": {
            "epoch_examples": 128,
            "maximum_per_example": [2, 2],
            "maximum_per_sequence": 32,
            "maximum_per_sequence_batch": 2,
            "minimum_coverage_before_patience": [0.5, 0.5],
        },
    }


def test_coverage_first_caps_actual_batches_and_resume_rng():
    data = {
        "sources": np.array([0] * 64 + [1] * 192),
        "origins": np.array([0] * 64 + [2] * 192),
        "sequences": np.arange(256) // 4,
        "groups": np.arange(256) % 4,
    }
    counts = torch.zeros(256, dtype=torch.int32)
    generator = torch.Generator().manual_seed(19)
    for iteration in range(4):
        rng = generator.get_state()
        order = coverage_order(data, counts, policy(), generator)
        restored = torch.Generator()
        restored.set_state(rng)
        assert torch.equal(order, coverage_order(data, counts, policy(), restored))
        assert len(order) == len(order.unique()) == 128
        for start in range(0, len(order), 16):
            assert max(Counter(data["sequences"][order[start : start + 16].numpy()]).values()) <= 2
        assert int(counts[order].max()) == iteration // 2
        counts[order] += 1
    assert counts.min() == counts.max() == 2
    assert len(coverage_order(data, counts, policy(), generator)) == 0
    assert exposure_summary(data, counts)["coverage"] == 1


def test_subthreshold_best_is_saved_without_resetting_patience(tmp_path, monkeypatch):
    from test_evaluator_training import corpus, make_dataset, setup

    data, config = setup(tmp_path)
    for split in ("train", "validation"):
        make_dataset(data, corpus()[:8] * 4, split)
        np.save(data / f"{split}-groups.npy", np.arange(32) % 4)
        np.save(data / f"{split}-sources.npy", np.array([0] * 8 + [1] * 24))
        np.save(data / f"{split}-origins.npy", np.array([0] * 8 + [2] * 24))
        np.save(data / f"{split}-sequences.npy", np.arange(32))
    config.update(
        policy(),
        max_steps=3,
        max_epochs=5,
        validation_every=1,
        patience=10,
        minimum_relative_improvement=0.002,
        maximum_replay_regression_ratio=1.03,
    )
    config["coverage_sampler"]["epoch_examples"] = 32
    config["coverage_sampler"]["minimum_coverage_before_patience"] = [0, 0]
    calls = 0

    def metrics(*args, **kwargs):
        nonlocal calls
        calls += 1
        return {
            "positions": 24,
            "loss": 1 - (calls - 1) * 0.0005,
            "cp_mae": 1,
            "cp_rmse": 1,
            "overestimate_above_600cp_rate": 0,
        }

    monkeypatch.setattr(training, "evaluate", metrics)
    monkeypatch.setattr(
        training,
        "evaluate_groups",
        lambda *a, **k: {
            group: {
                "positions": 2,
                "loss": 1,
                "cp_mae": 1,
                "cp_rmse": 1,
                "overestimate_above_600cp_rate": 0,
            }
            for group in training.GROUPS
        },
    )
    result = training.train(data, tmp_path / "fit", config, {"fixture": "c2"})
    assert result["best_step"] == 3
    assert result["history"][-1]["observed_best_saved"]
    assert not result["history"][-1]["meaningful_improvement"]
    assert result["history"][-1]["stale_intervals"] == 3
    assert result["best_exposure"]["exposures"] == result["example_exposures"]
    ref = json.loads((tmp_path / "fit/resume.json").read_text())
    saved = torch.load(tmp_path / "fit" / ref["path"], weights_only=True)
    assert saved["patience_loss"] == 1


def test_packed_move_flags_and_bad_positions():
    assert packed_move(3475) == "4a3b"
    assert packed_move(34108) == "2b7g+"
    assert packed_move(16384 + (1 << 7) + 40) == "P*5e"
    for move in (0, 127, 16384 + 32768 + (1 << 7) + 40):
        with pytest.raises(ValueError):
            packed_move(move)
    for raw in (b"", bytes(32), bytes([255]) * 32):
        with pytest.raises(ValueError):
            unpack_position(raw)


def test_psv_initial_position_decodes_square_order_and_turn():
    # Construct the standard board ourselves, rather than redistribute a teacher row.
    bits = []

    def write(value, count):
        bits.extend((value >> n) & 1 for n in range(count))

    write(1, 1)
    write(44, 7)  # 5i black king, 5a white king
    write(36, 7)
    codes = {
        "": (0, 1),
        "P": (1, 2),
        "L": (3, 4),
        "N": (11, 4),
        "S": (7, 4),
        "G": (15, 5),
        "B": (31, 6),
        "R": (63, 6),
    }
    ranks = [
        "lnsgkgsnl",
        ".r.....b.",
        "ppppppppp",
        ".........",
        ".........",
        ".........",
        "PPPPPPPPP",
        ".B.....R.",
        "LNSGKGSNL",
    ]
    for file in range(1, 10):
        for rank in range(9):
            name = ranks[rank][9 - file]
            if name.upper() == "K":
                continue
            piece = "" if name == "." else name.upper()
            write(*codes[piece])
            if piece:
                if piece != "G":
                    write(0, 1)
                write(int(name.islower()), 1)
    assert len(bits) == 256
    raw = sum(bit << n for n, bit in enumerate(bits)).to_bytes(32, "little")
    from open_shogi_training.evaluator_data import START

    assert unpack_position(raw, 77) == START.replace(" b - 1", " w - 77")


def test_failed_registration_restores_existing_descriptor(tmp_path, monkeypatch):
    import subprocess

    from open_shogi_training import evaluator_development as development
    from open_shogi_training import evaluator_run as runner
    from open_shogi_training.evaluator_data import digest

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    model = tmp_path / "local/model.osaval03"
    model.parent.mkdir()
    model.write_bytes(b"model")
    descriptor = tmp_path / "local/core-prototype/r4c2.json"
    descriptor.parent.mkdir()
    descriptor.write_bytes(b"previous exact bytes")
    for path in runner.RUNTIME_SOURCES.values():
        file = tmp_path / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"runtime")
    config = {
        "run_id": "fixture",
        "development_integration": {"port": 5175, "ui_identity": {"files": {}}},
        "runtime": {
            k: {"path": p, "sha256": digest(tmp_path / p)}
            for k, p in runner.RUNTIME_SOURCES.items()
        },
    }
    output = tmp_path / "local/proof"
    from test_evaluator_run import audit_report, put

    put(tmp_path / "scripts/check_evaluator_model.mjs")

    def command(args, **kwargs):
        if "check_evaluator_model.mjs" in args[1]:
            audit_report(
                tmp_path,
                tmp_path / "local",
                config,
                model=model,
                report_path=output / "model-audit.json",
            )
        else:
            raise subprocess.CalledProcessError(1, args)

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(development.subprocess, "run", command)
    monkeypatch.setattr(development.urllib.request, "urlopen", lambda *a, **k: Response())
    with pytest.raises(subprocess.CalledProcessError):
        development.register(tmp_path, tmp_path / "local", config, model, output)
    assert descriptor.read_bytes() == b"previous exact bytes"
    assert json.loads((output / "rollback.json").read_text())["restored_previous"]


def test_every_c2_stage_has_progress_monitoring(tmp_path):
    from open_shogi_training.evaluator_run import ALL_STAGES, _progress_signature

    for stage in ALL_STAGES:
        assert isinstance(_progress_signature(tmp_path, stage), tuple)


ORIGIN_URL = "https://huggingface.co/datasets/nodchip/shogi_hao_depth9/resolve/main/shard.bin"


def _redirect(handler, newurl, hops=0):
    request = urllib.request.Request(ORIGIN_URL)
    request.reviewed_redirect_hops = hops
    return handler.redirect_request(request, None, 302, "Found", {}, newurl)


def test_redirect_handler_validates_every_intermediate_hop():
    assert any(
        isinstance(handler, r4_sources._ReviewedHostRedirectHandler)
        for handler in r4_sources._OPENER.handlers
    )
    handler = r4_sources._ReviewedHostRedirectHandler()
    followed = _redirect(handler, "https://cdn-lfs.huggingface.co/repo/shard.bin")
    assert followed.full_url == "https://cdn-lfs.huggingface.co/repo/shard.bin"
    assert followed.get_method() == "GET"
    assert followed.reviewed_redirect_hops == 1
    for newurl in (
        "https://cdn-lfs.huggingface.co/repo/shard.bin",
        "https://cas-bridge.xethub.hf.co/repo/shard.bin",
    ):
        assert _redirect(handler, newurl).full_url == newurl
    for newurl in (
        "http://huggingface.co/repo/shard.bin",
        "https://evil.example.com/repo/shard.bin",
        "https://xhuggingface.co/repo/shard.bin",
        "https://huggingface.co.evil.com/repo/shard.bin",
        "https://hf.co.evil.com/repo/shard.bin",
        "https://hf.co/repo/shard.bin",
        "https://huggingface.co./repo/shard.bin",
        "https://user:pass@huggingface.co/repo/shard.bin",
    ):
        with pytest.raises(ValueError):
            _redirect(handler, newurl)


def test_redirect_handler_enforces_strict_hop_limit():
    handler = r4_sources._ReviewedHostRedirectHandler()
    request = urllib.request.Request(ORIGIN_URL)
    for hops in range(1, handler.maximum_hops + 1):
        request = handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://cas-bridge.xethub.hf.co/repo/shard.bin",
        )
        assert request.reviewed_redirect_hops == hops
    with pytest.raises(ValueError, match="hop limit"):
        handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://cas-bridge.xethub.hf.co/repo/shard.bin",
        )
