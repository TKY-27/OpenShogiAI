from __future__ import annotations

import hashlib
import struct

import numpy as np
import pytest
from open_shogi_training.phase10r_model import parse_sfen
from open_shogi_training.phase10t_model import (
    FEATURE_COUNT,
    Phase10TExample,
    Phase10TModel,
    Phase10TModelError,
    feature_ids,
    train_random_lineage,
    train_supervised_lineage,
)


def test_a1_feature_schema_and_random_round_trip() -> None:
    sfen = "4k4/9/9/9/4P4/9/9/9/4K4 b - 1"
    position = parse_sfen(sfen)
    black = feature_ids(position, 0)
    white = feature_ids(position, 1)
    assert FEATURE_COUNT == 8433
    assert black == tuple(sorted(set(black)))
    assert white == tuple(sorted(set(white)))
    model = Phase10TModel.random(20260907)
    score, logits = model.evaluate(sfen)
    assert np.isfinite(score)
    assert logits.shape == (3,)
    restored = Phase10TModel.from_bytes(model.to_bytes())
    assert restored.sha256 == model.sha256
    assert restored.evaluate(sfen)[0] == pytest.approx(score, abs=1e-5)


def test_a1_parser_rejects_corrupt_payload() -> None:
    data = bytearray(Phase10TModel.random(20260907).to_bytes())
    data[-1] ^= 1
    with pytest.raises(Phase10TModelError, match="checksum"):
        Phase10TModel.from_bytes(bytes(data))


def test_checksum_valid_extreme_finite_parameter_rejects_before_inference() -> None:
    data = bytearray(Phase10TModel.random(20260907).to_bytes())
    struct.pack_into("<f", data, 44, float(np.finfo(np.float32).max))
    data[-32:] = hashlib.sha256(data[44:-32]).digest()
    with pytest.raises(Phase10TModelError, match="magnitude"):
        Phase10TModel.from_bytes(bytes(data))


def test_micro_training_is_random_lineage_and_bounded() -> None:
    rows = [
        Phase10TExample(
            "4k4/9/9/9/4P4/9/9/9/4K4 b - 1",
            120.0,
            2,
        ),
        Phase10TExample(
            "4k4/9/9/9/4P4/9/9/9/4K4 w - 1",
            -120.0,
            0,
        ),
    ]
    model, stats = train_random_lineage(rows, seed=20260907, max_steps=2, time_limit_seconds=10)
    assert stats["random_initialization"] is True
    assert stats["steps"] == 2
    model.validate()


def test_supervised_training_keeps_source_macro_and_validation_lineage() -> None:
    sfen = "4k4/9/9/9/4P4/9/9/9/4K4 b - 1"
    rows = [
        Phase10TExample(
            sfen,
            float(index * 20 - 100),
            index % 3,
            source="aobazero" if index % 2 else "wcsc",
        )
        for index in range(20)
    ]
    pre, selected, last, stats = train_supervised_lineage(
        rows,
        seed=20260907,
        max_passes=1,
        batch_size=4,
        validation_every_steps=1,
        patience=2,
    )
    assert stats["random_initialization"] is True
    assert stats["source_macro_weighting"] is True
    assert stats["validation_examples"] == 2
    assert stats["steps"] > 0
    assert pre.sha256 != selected.sha256
    assert last.sha256 != pre.sha256
