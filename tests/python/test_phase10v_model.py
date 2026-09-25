"""OSAVAL03 independent reference, calibration and all-layer learning gates."""

import hashlib
from dataclasses import replace

import numpy as np
import pytest
import torch
from open_shogi_training.phase10v_model import (
    FEATURE_COUNT,
    STM_OFFSET,
    Phase10VModel,
    Phase10VModelError,
    _train_verified_streams,
    micro_overfit,
    sparse_features,
    torch_forward,
    torch_parameters,
    training_loss,
)
from open_shogi_training.phase10v_targets import CandidateTarget, Phase10VExample, TeacherScore

START = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
CHILD_A = "lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 2"
CHILD_B = "lnsgkgsnl/1r5b1/ppppppppp/9/9/7P1/PPPPPPP1P/1B5R1/LNSGKGSNL w - 2"


@pytest.mark.parametrize("width", [256, 512])
def test_osaval03_roundtrip_and_forward_parity(width):
    model = Phase10VModel.random(71, width)
    data = model.to_bytes()
    assert data[:8] == b"OSAVAL03"
    assert data[-32:] == hashlib.sha256(data[:-32]).digest()
    restored = Phase10VModel.from_bytes(data)
    assert restored.to_bytes() == data
    with torch.no_grad():
        actual = torch_forward(torch_parameters(model), [START, CHILD_A, CHILD_B]).numpy()
    for index, sfen in enumerate([START, CHILD_A, CHILD_B]):
        cp, wdl = model.evaluate(sfen)
        np.testing.assert_allclose(actual[index], np.r_[cp, wdl], atol=0.0001, rtol=0.0001)


@pytest.mark.parametrize("offset", [32, 44, -1])
def test_header_seed_payload_and_checksum_corruption_fail_closed(offset):
    data = bytearray(Phase10VModel.random().to_bytes())
    data[offset] ^= 1
    with pytest.raises(Phase10VModelError):
        Phase10VModel.from_bytes(bytes(data))


def test_nonfinite_and_invalid_lengths_rejected():
    model = Phase10VModel.random()
    data = model.to_bytes()
    for malformed in (data[:-1], data + b"\0", b"OSAVAL02"):
        with pytest.raises(Phase10VModelError):
            Phase10VModel.from_bytes(malformed)
    model.table[0, 0] = np.nan
    with pytest.raises(Phase10VModelError):
        model.to_bytes()


def test_features_stm_and_move_number_semantics():
    black, white, _ = sparse_features(START)
    changed_black, changed_white, _ = sparse_features(START.replace(" b ", " w "))
    assert set(black) ^ set(changed_black) == {STM_OFFSET, STM_OFFSET + 1}
    assert set(white) ^ set(changed_white) == {STM_OFFSET, STM_OFFSET + 1}
    assert sparse_features(START) == sparse_features(START[:-1] + "999")
    assert max(black + white) < FEATURE_COUNT
    # Color swap + board rotation leaves the relative features equal at symmetric start.
    assert set(black) - {STM_OFFSET} == set(white) - {STM_OFFSET + 1}


def test_direct_cp_rounding_clamp_and_no_wdl_conversion():
    model = Phase10VModel.random()
    model.head_weight.fill(0)
    for cp, expected in [(1.5, 2), (-1.5, -2), (25000, 20000), (-25000, -20000)]:
        model.head_bias[:] = [cp, 999, -999, 500]
        assert model.search_score(START) == expected


def test_all_layers_have_cp_and_ranking_gradients():
    torch.set_num_threads(1)
    model = Phase10VModel.random()
    parameters = torch_parameters(model)
    candidates = (
        CandidateTarget("7g7f", CHILD_A, TeacherScore("cp", 400)),
        CandidateTarget("2g2f", CHILD_B, TeacherScore("cp", -200)),
    )
    row = Phase10VExample(START, TeacherScore("cp", 700), 2, candidates)
    loss, metrics = training_loss(parameters, [row])
    loss.backward()
    assert metrics["ranking_loss"] > 0
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    # Ranking alone flows through child features and the scalar head, not detached predictions.
    parameters = torch_parameters(model)
    ranking_only = replace(row, score=None, wdl=None)
    loss, _ = training_loss(parameters, [ranking_only])
    loss.backward()
    assert parameters[0].grad.abs().sum() > 0
    assert parameters[4].grad[:, 0].abs().sum() > 0
    assert parameters[4].grad[:, 1:].abs().sum() == 0


def test_mate_and_missing_candidates_masked():
    parameters = torch_parameters(Phase10VModel.random())
    row = Phase10VExample(START, TeacherScore("mate", 7))
    row.validate(production=False)
    loss, metrics = training_loss(parameters, [row])
    loss.backward()
    assert metrics["cp_loss"] == 0
    assert metrics["ranking_loss"] == 0
    assert all(parameter.grad.abs().sum() == 0 for parameter in parameters)


def test_bounded_micro_overfit_uses_real_cp_units():
    rows = [
        Phase10VExample(START, TeacherScore("cp", 120)),
        Phase10VExample(CHILD_A, TeacherScore("cp", -90)),
    ]
    _, metrics = micro_overfit(rows, steps=100)
    assert metrics["final_loss"] < metrics["initial_loss"] * 0.2
    assert metrics["cp_mae"] < 25
    assert metrics["all_layers_have_gradient"]


def test_streaming_training_rejects_game_and_position_leakage(tmp_path):
    train = Phase10VExample(
        START,
        TeacherScore("cp", 120),
        source="approved",
        source_game_id="game",
        provenance_sha256="a" * 64,
    )
    for validation in (
        replace(train, split="validation"),
        replace(train, sfen=CHILD_A, split="validation"),
    ):
        with pytest.raises(ValueError, match="leakage"):
            _train_verified_streams(
                lambda: iter([train]),
                lambda validation=validation: iter([validation]),
                tmp_path / "out",
            )
    assert not (tmp_path / "out").exists()


def test_streaming_tiny_training_deterministic_checkpoints(tmp_path):
    train = Phase10VExample(
        START,
        TeacherScore("cp", 120),
        source="approved",
        source_game_id="train-game",
        provenance_sha256="a" * 64,
    )
    validation = replace(train, sfen=CHILD_A, split="validation", source_game_id="dev-game")
    results = []
    for name in ("first", "second"):
        results.append(
            _train_verified_streams(
                lambda: iter([train]),
                lambda: iter([validation]),
                tmp_path / name,
                max_steps=2,
                max_passes=2,
                batch_size=1,
                validation_every_steps=1,
            )
        )
    assert results[0] == results[1]
    for filename in ("best.osaval03", "latest.osaval03"):
        assert (tmp_path / "first" / filename).read_bytes() == (
            tmp_path / "second" / filename
        ).read_bytes()


def test_source_macro_loss_does_not_overweight_duplicate_source_rows():
    parameters = torch_parameters(Phase10VModel.random())
    first = Phase10VExample(START, TeacherScore("cp", 1500), source="one")
    second = Phase10VExample(CHILD_A, TeacherScore("cp", -1300), source="two")
    baseline, _ = training_loss(parameters, [first, second])
    repeated, _ = training_loss(parameters, [first] * 5 + [second])
    assert float(baseline.detach()) == pytest.approx(float(repeated.detach()), abs=1e-6)


def test_production_training_calls_raw_evidence_verifier_before_reading(monkeypatch, tmp_path):
    from open_shogi_training import phase10v_data
    from open_shogi_training.phase10v_model import train_supervised

    def reject(*args):
        raise ValueError("raw teacher evidence does not bind this dataset")

    monkeypatch.setattr(phase10v_data, "verify_training_inputs", reject, raising=False)
    # The 80 GiB production floor is an environmental guard, not the subject
    # here; neutralize it so the assertion below holds on small disks too.
    from open_shogi_training import phase10v_model

    monkeypatch.setattr(
        phase10v_model,
        "_ensure_free_space",
        lambda path: None,
        raising=False,
    )
    with pytest.raises(ValueError, match="raw teacher evidence"):
        train_supervised(
            tmp_path / "nonexistent-train",
            tmp_path / "nonexistent-validation",
            tmp_path / "out",
            data_receipt_path=tmp_path / "receipt",
            data_receipt_sha256="a" * 64,
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("field_offset,value", [(8, 4), (12, 2), (16, 8426), (20, 128)])
def test_rechecksummed_incompatible_header_still_fails(field_offset, value):
    import struct

    data = bytearray(Phase10VModel.random().to_bytes())
    struct.pack_into("<I", data, field_offset, value)
    data[-32:] = hashlib.sha256(data[:-32]).digest()
    with pytest.raises(Phase10VModelError):
        Phase10VModel.from_bytes(bytes(data))


def test_conditioned_trained_parameters_export_in_actual_cp_units():
    from open_shogi_training.phase10v_model import _snapshot

    parameters = torch_parameters(Phase10VModel.random())
    with torch.no_grad():
        parameters[4][:, 0].add_(0.4)
        parameters[5][0].add_(2.0)
        expected = torch_forward(parameters, [START, CHILD_A]).numpy()
    exported = _snapshot(parameters, 20260908)
    for index, sfen in enumerate([START, CHILD_A]):
        cp, wdl = exported.evaluate(sfen)
        np.testing.assert_allclose(np.r_[cp, wdl], expected[index], atol=0.001, rtol=0.00001)


def test_offgrid_accumulator_rejected_even_with_valid_checksum():
    import struct

    data = bytearray(Phase10VModel.random().to_bytes())
    struct.pack_into("<f", data, 44, 0.01)
    data[-32:] = hashlib.sha256(data[:-32]).digest()
    with pytest.raises(Phase10VModelError, match="Q20"):
        Phase10VModel.from_bytes(bytes(data))


def test_q20_double_accumulator_preserves_tiny_bias_after_large_make_unmake():
    from open_shogi_training.phase10v_model import ACCUMULATOR_GRID

    model = Phase10VModel.random()
    model.table.fill(0)
    model.bias.fill(0)
    model.bias[0] = round(0.01 * ACCUMULATOR_GRID) / ACCUMULATOR_GRID
    model.table[0, 0] = 1_000_000
    model.table[1, 0] = -1_000_000
    root = model.accumulate([0, 1])
    incremental = root.copy()
    for _ in range(1000):
        incremental -= model.table[0]
        incremental += model.table[0]
    np.testing.assert_array_equal(incremental, root)
    np.testing.assert_array_equal(root, model.bias.astype(np.float64))
    incremental -= model.table[0]
    np.testing.assert_array_equal(incremental, model.accumulate([1]))


def test_initial_checkpoint_requires_pinned_random_training_lineage(tmp_path):
    import json

    from open_shogi_training.phase10v_model import _verify_initial_lineage

    model = Phase10VModel.random()
    with pytest.raises(ValueError, match="pinned parent"):
        _verify_initial_lineage(model, model.seed, model.width, None, None, None)
    receipt = {
        "schema": "open_shogiai_phase10v_training/v1",
        "seed": model.seed,
        "width": model.width,
        "best_sha256": model.sha256,
        "latest_sha256": model.sha256,
        "lineage_root_sha256": model.sha256,
        "verified_data_receipt": {"sha256": "a" * 64},
    }
    path = tmp_path / "parent.json"
    path.write_text(json.dumps(receipt))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert _verify_initial_lineage(model, model.seed, model.width, model.sha256, path, digest)
    with pytest.raises(ValueError, match="hash mismatch"):
        _verify_initial_lineage(model, model.seed, model.width, model.sha256, path, "b" * 64)
    receipt["lineage_root_sha256"] = "c" * 64
    path.write_text(json.dumps(receipt))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="random-initialized"):
        _verify_initial_lineage(model, model.seed, model.width, model.sha256, path, digest)


def test_bounded_affine_calibration_fit_and_sign_guard():
    from open_shogi_training.phase10v_model import fit_affine_calibration

    assert fit_affine_calibration([-100, 0, 100], [-170, 30, 230]) == (2, 30)
    assert fit_affine_calibration([-100, 0, 100], [-10000, 0, 10000]) == (4, 0)
    assert fit_affine_calibration([-100, 0, 100], [9900, 10000, 10100]) == (1, 2000)
    for predictions, targets in [
        ([0, 0, 0], [1, 2, 3]),
        ([-1, 0, 1], [1, 0, -1]),
        ([-1, 0, float("nan")], [1, 2, 3]),
    ]:
        with pytest.raises(ValueError):
            fit_affine_calibration(predictions, targets)


def test_affine_candidate_changes_only_cp_head():
    from open_shogi_training.phase10v_model import affine_candidate

    model = Phase10VModel.random()
    candidate = affine_candidate(model, 2, 30)
    for sfen in (START, CHILD_A, CHILD_B):
        cp, wdl = model.evaluate(sfen)
        transformed_cp, transformed_wdl = candidate.evaluate(sfen)
        assert transformed_cp == pytest.approx(2 * cp + 30, abs=0.001)
        np.testing.assert_array_equal(wdl, transformed_wdl)
    for original, changed in zip(model.parameters[:4], candidate.parameters[:4], strict=True):
        np.testing.assert_array_equal(original, changed)
    for scale, offset in [(-1, 0), (5, 0), (1, 2001), (float("nan"), 0)]:
        with pytest.raises(ValueError):
            affine_candidate(model, scale, offset)


def test_calibration_never_reads_final_holdout_as_validation(monkeypatch, tmp_path):
    import json

    from open_shogi_training import phase10v_data, phase10v_model
    from open_shogi_training.phase10v_targets import TARGET_SCHEMA

    model = Phase10VModel.random()
    path = tmp_path / "model.osaval03"
    model.write(path)
    validation = tmp_path / "validation.jsonl"
    validation.write_text(json.dumps({"schema": TARGET_SCHEMA, "split": "final_holdout"}) + "\n")
    monkeypatch.setattr(phase10v_data, "verify_training_inputs", lambda *args: {"verified": True})
    monkeypatch.setattr(
        phase10v_model,
        "_verify_initial_lineage",
        lambda *args: {"lineage_root_sha256": model.sha256},
    )
    monkeypatch.setattr(phase10v_model, "_ensure_free_space", lambda path: None)
    with pytest.raises(ValueError, match="split mismatch"):
        phase10v_model.calibrate_candidate(
            path,
            tmp_path / "train.jsonl",
            validation,
            tmp_path / "out",
            expected_model_sha256=model.sha256,
            parent_receipt_path=tmp_path / "parent.json",
            parent_receipt_sha256="a" * 64,
            data_receipt_path=tmp_path / "data.json",
            data_receipt_sha256="b" * 64,
        )
    assert not (tmp_path / "out").exists()


def test_calibration_publishes_bounded_proposal_with_identity(monkeypatch, tmp_path):
    import json

    from open_shogi_training import phase10v_data, phase10v_model

    model = Phase10VModel.random()
    path = tmp_path / "model.osaval03"
    model.write(path)
    rows = [
        Phase10VExample(
            sfen, TeacherScore("cp", round(2 * model.evaluate(sfen)[0] + 30)), split="validation"
        )
        for sfen in (START, CHILD_A, CHILD_B)
    ]
    monkeypatch.setattr(
        phase10v_data, "verify_training_inputs", lambda *args: {"validation_sha256": "c" * 64}
    )
    monkeypatch.setattr(
        phase10v_model,
        "_verify_initial_lineage",
        lambda *args: {"lineage_root_sha256": model.sha256},
    )
    monkeypatch.setattr(phase10v_model, "_ensure_free_space", lambda path: None)

    def validation_only(path, *, expected_split):
        assert expected_split == "validation"
        return iter(rows)

    monkeypatch.setattr(phase10v_model, "iter_examples", validation_only)
    out = tmp_path / "out"
    metrics = phase10v_model.calibrate_candidate(
        path,
        tmp_path / "train.jsonl",
        tmp_path / "validation.jsonl",
        out,
        expected_model_sha256=model.sha256,
        parent_receipt_path=tmp_path / "parent.json",
        parent_receipt_sha256="a" * 64,
        data_receipt_path=tmp_path / "data.json",
        data_receipt_sha256="b" * 64,
        max_examples=3,
    )
    assert metrics["scale"] == pytest.approx(2, abs=0.03)
    assert metrics["after_cp_mae"] < metrics["before_cp_mae"]
    assert metrics["fit_split"] == "validation"
    assert metrics["cp_examples"] == 3
    assert "proposal_only" in metrics["selection"]
    assert Phase10VModel.read(out / "calibrated.osaval03").sha256 == metrics["candidate_sha256"]
    assert json.loads((out / "calibration.json").read_text()) == metrics
