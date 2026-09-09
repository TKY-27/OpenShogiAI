"""Source-game splitting and independent-test isolation for the small controller."""

import numpy as np
import pytest
from open_shogi_training.core_prototype import canonical, fit, partition


def test_move_number_does_not_hide_cross_split_duplicate():
    a = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
    assert canonical(a) == canonical(a[:-1] + "101")
    assert partition(15) == "train"
    assert partition(16) == "validation"
    assert partition(20) == "development_test"
    with pytest.raises(ValueError):
        partition(24)


def test_test_features_and_labels_never_change_normalization_or_selected_checkpoint():
    rng = np.random.default_rng(12)
    groups = {}
    for split in ("train", "validation", "development_test"):
        x = rng.normal(size=(120, 10))
        x[:, 0] = np.arange(120) % 2
        y = (x[:, 1] > 0).astype(float)
        groups[split] = [
            (a.tolist(), float(b), i // 10, str(i // 3))
            for i, (a, b) in enumerate(zip(x, y, strict=True))
        ]
    first, _, _ = fit(groups, "a" * 64, steps=20)
    groups["development_test"] = [
        ([a[0], *([100.0] * 9)], 1 - b, g, p) for a, b, g, p in groups["development_test"]
    ]
    second, _, _ = fit(groups, "a" * 64, steps=20)
    assert first == second


def test_more_epochs_cannot_hide_absent_teacher_signal():
    rows = [([float(i % 2), *([0.0] * 9)], 0.0, i, str(i)) for i in range(20)]
    groups = dict.fromkeys(("train", "validation", "development_test"), rows)
    with pytest.raises(ValueError, match="teacher signal"):
        fit(groups, "a" * 64)
    with pytest.raises(ValueError, match="capped"):
        fit(groups, "a" * 64, steps=301)
