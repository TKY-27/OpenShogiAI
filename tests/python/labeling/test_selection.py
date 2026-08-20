from __future__ import annotations

import json
from pathlib import Path

import pytest
from open_shogi_training.labeling.selection import SelectionError, select_positions

from .helpers import (
    make_fake_project,
    phase3_position,
    write_phase3_dataset,
)


def test_cross_split_dedup_uses_test_then_validation_then_train_priority(
    tmp_path: Path,
) -> None:
    config, _, _ = make_fake_project(tmp_path, max_positions=10)
    rows = [
        phase3_position(game="train", index=0, sfen="same b - 1", split="train"),
        phase3_position(game="validation", index=1, sfen="same b - 9", split="validation"),
        phase3_position(game="test", index=2, sfen="same b - 20", split="test"),
        phase3_position(game="unique", index=15, sfen="unique w - 16", split="train"),
    ]
    positions, manifest = write_phase3_dataset(tmp_path, rows)

    result = select_positions(positions, manifest, config.selection)

    assert len(result.positions) == 2
    assert {item.split for item in result.positions} == {"test", "train"}
    assert any(item.game_id == rows[2]["gameId"] for item in result.positions)
    assert all(item.game_id != rows[0]["gameId"] for item in result.positions)
    assert all(item.game_id != rows[1]["gameId"] for item in result.positions)
    assert result.cross_split_duplicates_excluded == 2
    assert result.unique_eligible_states == 2
    assert (
        result.selection_sha256
        == select_positions(positions, manifest, config.selection).selection_sha256
    )


def test_selection_round_robins_stages_and_games_with_a_per_game_cap(tmp_path: Path) -> None:
    config, _, _ = make_fake_project(tmp_path, max_positions=6)
    rows = []
    for game_index in range(3):
        for position_index in (0, 10, 25):
            rows.append(
                phase3_position(
                    game=f"game-{game_index}",
                    index=position_index,
                    sfen=f"state-{game_index}-{position_index} b - {position_index + 1}",
                    split=("test", "validation", "train")[game_index],
                )
            )
    positions, manifest = write_phase3_dataset(tmp_path, rows)

    result = select_positions(positions, manifest, config.selection)

    assert len(result.positions) == 6
    assert set(result.per_stage) == {"opening", "middlegame", "endgame"}
    assert all(count > 0 for count in result.per_stage.values())
    assert all(
        count <= config.selection.max_positions_per_game for count in result.per_game.values()
    )
    assert len({item.game_id for item in result.positions}) == 3


def test_selection_rejects_positions_not_bound_by_dataset_manifest(tmp_path: Path) -> None:
    config, _, _ = make_fake_project(tmp_path)
    positions, manifest = write_phase3_dataset(
        tmp_path,
        [phase3_position(game="game", index=0, sfen="state b - 1")],
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["artifacts"][positions.name]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SelectionError, match="does not match"):
        select_positions(positions, manifest, config.selection)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sourceId", "different-source"),
        ("outcome", "white_win"),
        ("sfen", "state b - 99"),
    ],
)
def test_v2_selection_digest_binds_exact_label_source_fields_while_v1_is_reproducible(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    config, _, _ = make_fake_project(tmp_path, max_positions=1)
    row = phase3_position(game="game", index=0, sfen="state b - 1")
    positions, manifest = write_phase3_dataset(tmp_path, [row])
    original = select_positions(positions, manifest, config.selection)

    changed = dict(row)
    changed[field] = value
    changed_root = tmp_path / "changed"
    changed_root.mkdir()
    positions, manifest = write_phase3_dataset(changed_root, [changed])
    revised = select_positions(positions, manifest, config.selection)

    assert revised.selection_sha256 != original.selection_sha256
    assert revised.legacy_selection_sha256 == original.legacy_selection_sha256
