from __future__ import annotations

import numpy as np
import pytest
from open_shogi_training.phase10r_model import HistoryFacts, parse_sfen
from open_shogi_training.phase10t_accumulator import (
    FEATURE_COUNT,
    HAND_OFFSET,
    HISTORY_OFFSET,
    TRANSITIONS,
    RandomAccumulator,
    feature_ids,
)


@pytest.fixture(scope="module")
def model() -> RandomAccumulator:
    return RandomAccumulator()


@pytest.mark.parametrize("name", TRANSITIONS)
def test_delta_refresh_and_exact_unmake(model: RandomAccumulator, name: str) -> None:
    before_sfen, after_sfen = TRANSITIONS[name]
    before = model.refresh(before_sfen)
    original = before.values.copy()
    update = model.update(before, after_sfen)
    expected = model.refresh(after_sfen)
    assert update.after.features == expected.features
    np.testing.assert_allclose(update.after.values, expected.values, rtol=0, atol=1e-7)
    np.testing.assert_array_equal(before.values, original)
    np.testing.assert_array_equal(model.unmake(update).values, original)
    assert update.rebuilt == ((True, False) if name == "king" else (False, False))
    np.testing.assert_allclose(
        model.evaluate(update.after)[1], model.evaluate(expected)[1], atol=1e-8
    )


def test_side_to_move_reorders_accumulators_only(model: RandomAccumulator) -> None:
    sfen = TRANSITIONS["quiet"][0]
    black = model.refresh(sfen)
    white = model.refresh(sfen.replace(" b ", " w "))
    np.testing.assert_array_equal(black.values, white.values)
    for state in (black, white):
        stm = state.position.side_to_move
        x = np.maximum(state.values[[stm, 1 - stm]].reshape(-1), 0)
        hidden = np.maximum(x @ model.hidden_weight + model.hidden_bias, 0)
        expected = hidden @ model.head_weight + model.head_bias
        score, logits = model.evaluate(state)
        assert score == expected[0]
        np.testing.assert_array_equal(logits, expected[1:])


def test_hand_unary_and_history_ownership() -> None:
    position = parse_sfen("4k4/9/9/9/9/9/9/9/4K4 b 2P 1")
    base = feature_ids(position, 0)
    assert {i for i in base if HAND_OFFSET <= i < HISTORY_OFFSET} == {HAND_OFFSET, HAND_OFFSET + 1}
    history = HistoryFacts(True, 3, True, False)
    own = feature_ids(position, 0, history)
    other = feature_ids(position, 1, history)
    assert HISTORY_OFFSET + 1 in own
    assert HISTORY_OFFSET + 4 in own
    assert HISTORY_OFFSET + 6 in own and HISTORY_OFFSET + 7 not in own
    assert HISTORY_OFFSET + 7 in other and HISTORY_OFFSET + 6 not in other
    assert max(own | other) < FEATURE_COUNT
    with pytest.raises(ValueError, match="unavailable history"):
        feature_ids(position, 0, HistoryFacts(False, 2))


def test_history_update_recompute(model: RandomAccumulator) -> None:
    sfen = TRANSITIONS["quiet"][0]
    before = model.refresh(sfen)
    history = HistoryFacts(True, 3, False, True)
    update = model.update(before, sfen, history)
    expected = model.refresh(sfen, history)
    np.testing.assert_allclose(update.after.values, expected.values, rtol=0, atol=1e-7)
    assert update.rebuilt == (False, False)


def test_zero_weights_have_no_material_fallback() -> None:
    model = RandomAccumulator()
    for value in vars(model).values():
        value.fill(0)
    for sfens in TRANSITIONS.values():
        for sfen in sfens:
            score, logits = model.evaluate(model.refresh(sfen))
            assert score == 0
            np.testing.assert_array_equal(logits, np.zeros(3))


def test_deterministic_random_initialization_and_costs(model: RandomAccumulator) -> None:
    np.testing.assert_array_equal(model.table, RandomAccumulator().table)
    assert model.costs()["features"] == 8433
    assert model.costs()["head_multiply_accumulates_per_leaf"] == 8320
    assert model.costs()["two_accumulator_float32_bytes"] == 1024


def test_color_rotation_symmetry(model: RandomAccumulator) -> None:
    black = model.refresh("4k4/9/9/9/3P5/9/9/9/4K4 b 2P 1")
    rotated = model.refresh("4k4/9/9/9/5p3/9/9/9/4K4 w 2p 1")
    assert black.features[0] == rotated.features[1]
    assert black.features[1] == rotated.features[0]
    assert model.evaluate(black)[0] == model.evaluate(rotated)[0]
    np.testing.assert_array_equal(model.evaluate(black)[1], model.evaluate(rotated)[1])


def test_white_king_rebuild_and_captured_promoted_piece_demotes(model: RandomAccumulator) -> None:
    before = model.refresh("4k4/9/9/9/4P4/9/9/9/4K4 w - 1")
    update = model.update(before, "9/4k4/9/9/4P4/9/9/9/4K4 b - 2")
    assert update.rebuilt == (False, True)
    expected = model.refresh("9/4k4/9/9/4P4/9/9/9/4K4 b - 2")
    np.testing.assert_allclose(update.after.values, expected.values, rtol=0, atol=1e-7)
    capture = model.update(
        model.refresh("4k4/9/9/4+p4/4P4/9/9/9/4K4 b - 1"),
        "4k4/9/9/4P4/9/9/9/9/4K4 w P 2",
    )
    assert HAND_OFFSET in capture.after.features[0]
    np.testing.assert_allclose(
        capture.after.values,
        model.refresh("4k4/9/9/4P4/9/9/9/9/4K4 w P 2").values,
        rtol=0,
        atol=1e-7,
    )
