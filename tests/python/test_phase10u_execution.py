from __future__ import annotations

from pathlib import Path

import pytest
from open_shogi_training.phase10t_model import HistoryFacts, Phase10TExample
from open_shogi_training.phase10u_execution import (
    ObservedTrainingRow,
    Phase10UExecutionError,
    _label_input_ref,
    split_by_component,
    successor_sfen,
    train_observed_a1,
)


def row(index: int, component: str, source: str = "aobazero") -> ObservedTrainingRow:
    root = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
    return ObservedTrainingRow(
        root=Phase10TExample(root, 0.0, None, HistoryFacts(), source),
        candidate_sfens=(successor_sfen(root, "7g7f"), successor_sfen(root, "2g2f")),
        ranking_pairs=((0, 1),),
        source=source,
        component_id=component,
        index=index,
    )


def test_successor_handles_board_move_drop_capture_and_promotion() -> None:
    initial = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
    assert successor_sfen(initial, "7g7f").endswith(" w - 2")
    assert successor_sfen("4k4/9/9/9/9/9/4R4/9/4K4 b P 1", "P*5e").endswith(" w - 2")
    captured = successor_sfen("4k4/9/9/9/9/9/4R4/4p4/4K4 b - 1", "5g5h")
    assert "P" in captured.split(" ")[2]
    promoted = successor_sfen("4k4/9/9/9/9/4P4/9/9/4K4 b - 1", "5f5e+")
    assert "+P" in promoted.split(" ")[0]


def test_successor_rejects_inconsistent_move() -> None:
    with pytest.raises(Phase10UExecutionError):
        successor_sfen("4k4/9/9/9/9/9/9/9/4K4 b - 1", "7g7f")


def test_component_split_never_splits_one_component() -> None:
    rows = [
        row(index, component)
        for index, component in enumerate(
            ("a", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k")
        )
    ]
    split = split_by_component(rows, seed=20260907)
    memberships = {}
    for name, values in (
        ("train", split.train),
        ("validation", split.validation),
        ("calibration", split.calibration),
    ):
        for value in values:
            memberships.setdefault(value.component_id, set()).add(name)
    assert all(len(names) == 1 for names in memberships.values())
    assert sum(split.component_counts.values()) == len(rows)


def test_component_split_rejects_empty_partition() -> None:
    with pytest.raises(Phase10UExecutionError):
        split_by_component([row(0, "only")], seed=20260907)


def test_observed_training_has_finite_ranking_validation() -> None:
    rows = [row(index, f"component-{index}") for index in range(20)]
    split = split_by_component(rows, seed=20260907)
    _, selected, _, stats = train_observed_a1(
        split,
        seed=20260907,
        max_passes=1,
        batch_size=4,
        validation_every_steps=1,
        patience=2,
    )
    assert selected.validate() is None
    assert stats["ranking_pairs_validation"] > 0
    assert stats["validation_history"]
    assert 0.0 <= stats["validation_history"][-1]["ranking_accuracy"] <= 1.0


def test_label_input_ref_normalizes_cli_relative_paths(tmp_path: Path) -> None:
    labels = tmp_path / "labels.jsonl"
    labels.write_text("x\n", encoding="utf-8")
    relative, resolved = _label_input_ref(tmp_path, Path("labels.jsonl"))
    assert relative == Path("labels.jsonl")
    assert resolved == labels
    absolute_relative, absolute_resolved = _label_input_ref(tmp_path, labels)
    assert absolute_relative == relative
    assert absolute_resolved == resolved
    with pytest.raises(Phase10UExecutionError):
        _label_input_ref(tmp_path, tmp_path.parent / "outside.jsonl")
