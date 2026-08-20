from dataclasses import replace
from pathlib import Path

import pytest
from open_shogi_training.models.config import load_feature_config
from open_shogi_training.models.features import (
    PIECE_KINDS,
    SQUARES,
    extract_features,
    feature_flags,
    feature_groups,
    feature_schema,
    input_dimension,
    parse_canonical_sfen,
    zero_feature_group,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STARTPOS = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"


def _default_config():
    return load_feature_config(PROJECT_ROOT / "configs/features/value_v0.toml")


def test_default_schema_has_exact_dimension_offsets_and_flags() -> None:
    config = _default_config()

    assert input_dimension(config) == 2 * 14 * 81 + 2 * 7 + 1 + 4 == 2287
    assert [(group.name, group.offset, group.dimension) for group in feature_groups(config)] == [
        ("board_planes", 0, 2268),
        ("hand_counts", 2268, 14),
        ("side_to_move", 2282, 1),
        ("king_coordinates", 2283, 4),
    ]
    assert feature_flags(config) == 0b01111
    schema = feature_schema(config)
    assert schema["outputPerspective"] == "current_side_to_move"
    assert schema["boardPlanes"]["pieceKindOrder"] == list(PIECE_KINDS)


def test_startpos_features_use_absolute_color_and_canonical_square_order() -> None:
    config = _default_config()
    features = extract_features(STARTPOS, config)

    black_pawn_plane = 0
    white_pawn_plane = len(PIECE_KINDS)
    assert features[black_pawn_plane * SQUARES + 6 * 9] == 1.0
    assert features[white_pawn_plane * SQUARES + 2 * 9] == 1.0
    assert sum(features[: 2 * len(PIECE_KINDS) * SQUARES]) == 40.0
    assert features[2282] == 1.0
    assert features[2283:2287] == [0.0, 1.0, 0.0, -1.0]


def test_white_to_move_changes_only_explicit_side_feature() -> None:
    config = _default_config()
    black = extract_features(STARTPOS, config)
    white = extract_features(STARTPOS.replace(" b - 1", " w - 1"), config)

    differing = [
        index for index, pair in enumerate(zip(black, white, strict=True)) if pair[0] != pair[1]
    ]
    assert differing == [2282]
    assert black[2282] == 1.0
    assert white[2282] == -1.0


def test_hands_are_normalized_by_physical_inventory() -> None:
    config = replace(
        _default_config(),
        board_planes=False,
        side_to_move=False,
        king_coordinates=False,
    )
    features = extract_features("4k4/9/9/9/9/9/9/9/4K4 b R2Pp 1", config)

    assert features == [
        2 / 18,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1 / 2,
        1 / 18,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]


def test_pseudo_attacks_include_first_occupied_ray_square() -> None:
    config = replace(
        _default_config(),
        board_planes=False,
        hand_counts=False,
        side_to_move=False,
        king_coordinates=False,
        pseudo_attacks=True,
    )
    # Black rook on 5i is blocked by its bishop on 5h. The friendly occupied square is
    # controlled, and the rook ray does not continue through it.
    sfen = "4k4/9/9/9/9/9/9/4B4/K3R4 b - 1"
    attacks = extract_features(sfen, config)

    assert len(attacks) == 162
    assert attacks[7 * 9 + 4] == 1.0
    assert attacks[6 * 9 + 4] == 0.0
    assert attacks[8 * 9 + 4] == 0.0


@pytest.mark.parametrize(
    "sfen,match",
    [
        ("4k4/9/9/9/9/9/9/9/4K4  b - 1", "exactly four"),
        ("4k4/9/9/9/9/9/9/9/4K4 b P2P 1", "duplicated"),
        ("4k4/9/9/9/9/9/9/9/4K4 b PR 1", "canonical"),
        ("4k4/9/9/9/9/9/P8/9/P3K4 b - 1", "two unpromoted pawns"),
        ("P3k4/9/9/9/9/9/9/9/4K4 b - 1", "dead unpromoted"),
        ("4k4/9/9/9/9/9/9/9/9 b - 1", "exactly one king"),
        ("4k4/9/9/9/9/9/9/9/4K4 b - 01", "canonical positive"),
    ],
)
def test_parser_rejects_noncanonical_or_invalid_sfen(sfen: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_canonical_sfen(sfen)


def test_zero_feature_group_preserves_dimension_and_other_groups() -> None:
    config = _default_config()
    original = extract_features(STARTPOS, config)
    ablated = zero_feature_group(original, config, "side_to_move")

    assert len(ablated) == len(original)
    assert ablated[2282] == 0.0
    assert ablated[:2282] == original[:2282]
    assert ablated[2283:] == original[2283:]
