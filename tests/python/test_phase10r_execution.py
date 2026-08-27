from __future__ import annotations

from pathlib import Path

import pytest
from open_shogi_training.phase10r_execution import (
    _example_row,
    _outcome_wdl,
    _ReplayFeatureProcess,
    _stream_rows,
)

ROOT = Path(__file__).resolve().parents[2]
START_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"


def test_outcome_wdl_uses_current_side_to_move_perspective() -> None:
    assert _outcome_wdl("black_win", "black") == 2
    assert _outcome_wdl("black_win", "white") == 0
    assert _outcome_wdl("white_win", "black") == 0
    assert _outcome_wdl("white_win", "white") == 2
    assert _outcome_wdl("draw", "black") == 1
    assert _outcome_wdl("unknown", "white") is None


def test_stream_rows_is_deterministic_and_applies_frozen_weights(tmp_path: Path) -> None:
    paths = {}
    for source in ("aobazero", "wcsc", "denryu"):
        path = tmp_path / f"{source}.jsonl"
        path.write_text(
            f'{{"source":"{source}","raw_targets":{{"source_row":1}}}}\n',
            encoding="utf-8",
        )
        paths[source] = path

    first = list(_stream_rows(paths, 10, seed=123))
    second = list(_stream_rows(paths, 10, seed=123))

    assert first == second
    assert [row["source"] for row in first].count("aobazero") == 4
    assert [row["source"] for row in first].count("wcsc") == 4
    assert [row["source"] for row in first].count("denryu") == 2
    assert [row["raw_targets"]["stream_index"] for row in first] == list(range(10))
    assert all(row["raw_targets"]["sampling_seed"] == 123 for row in first)


def test_replay_feature_helper_emits_authoritative_legal_root_and_history() -> None:
    with _ReplayFeatureProcess(ROOT) as helper:
        features = helper.game(START_SFEN, ("7g7f", "3c3d"))

    assert len(features) == 3
    assert features[0]["positionIndex"] == 0
    assert "7g7f" in features[0]["legalMoves"]
    assert features[0]["history"] == {
        "available": True,
        "repetitionCount": 1,
        "continuousCheckByUs": False,
        "continuousCheckByThem": False,
    }
    assert features[2]["sfen"].endswith(" b - 3")


def test_replay_feature_helper_rejects_an_illegal_replay() -> None:
    with (
        _ReplayFeatureProcess(ROOT) as helper,
        pytest.raises(RuntimeError, match="illegal USI move"),
    ):
        helper.game(START_SFEN, ("7g7f", "7g7f"))


def test_example_row_keeps_rust_legality_and_source_targets() -> None:
    row = {
        "artifact_id": "aobazero-no-noise-exact100",
        "source_id": "aobazero",
        "source_game_id": "game-1",
        "game_id": "normalized-game-1",
        "record_id": "record-1",
        "position_index": 0,
        "canonical_position_id": "position-1",
        "canonical_sfen": START_SFEN,
        "history_id": "history-1",
        "transposition_key": "transposition-1",
        "side_to_move": "black",
        "split": "train",
        "archive_member_path": "game.csa",
        "archive_member_sha256": "a" * 64,
        "record_sha256": "b" * 64,
        "parser_version": "parser/v1",
        "normalization_version": "normalization/v2",
    }
    payload = {
        "outcome": "black_win",
        "terminal": "TORYO",
        "usi_moves": ["7g7f", "3c3d"],
    }
    with _ReplayFeatureProcess(ROOT) as helper:
        features = helper.game(START_SFEN, payload["usi_moves"])
    result = _example_row(row, payload, features[0])

    assert result["schema"] == "phase10r_training_example/v1"
    assert result["played_move"] == "7g7f"
    assert result["wdl"] == 2
    assert result["wdl_mask"] is True
    assert result["raw_targets"]["legal_mask_runtime"].startswith("open-shogi-core/")


def test_terminal_replay_root_does_not_create_a_policy_target() -> None:
    row = {
        "artifact_id": "aobazero-no-noise-exact100",
        "source_id": "aobazero",
        "source_game_id": "game-1",
        "game_id": "normalized-game-1",
        "record_id": "record-1",
        "position_index": 0,
        "canonical_position_id": "position-1",
        "canonical_sfen": START_SFEN,
        "history_id": "history-1",
        "transposition_key": "transposition-1",
        "side_to_move": "black",
        "split": "train",
        "archive_member_path": "game.csa",
        "archive_member_sha256": "a" * 64,
        "record_sha256": "b" * 64,
        "parser_version": "parser/v1",
        "normalization_version": "normalization/v2",
    }
    feature = {
        "event": "position",
        "positionIndex": 0,
        "sfen": START_SFEN,
        "legalMoves": [],
        "history": {
            "available": True,
            "repetitionCount": 1,
            "continuousCheckByUs": False,
            "continuousCheckByThem": False,
        },
    }
    result = _example_row(
        row,
        {"outcome": "black_win", "terminal": "CHECKMATE", "usi_moves": []},
        feature,
    )

    assert result["legal_moves"] is None
    assert result["played_move"] is None
    assert result["raw_targets"]["terminal_position"] is True
