"""Deterministic, versioned value_v0 feature extraction from canonical SFEN."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from open_shogi_training.models.config import FeatureConfig, config_sha256

FEATURE_SCHEMA_VERSION = 1
BOARD_SIZE = 9
SQUARES = BOARD_SIZE * BOARD_SIZE
SIDES = ("black", "white")
PIECE_KINDS = (
    "P",
    "L",
    "N",
    "S",
    "G",
    "B",
    "R",
    "K",
    "+P",
    "+L",
    "+N",
    "+S",
    "+B",
    "+R",
)
HAND_PIECES = ("P", "L", "N", "S", "G", "B", "R")
HAND_MAXIMUMS = (18, 4, 4, 4, 4, 2, 2)
SFEN_HAND_ORDER = ("R", "B", "G", "S", "N", "L", "P")

FEATURE_BOARD = 1 << 0
FEATURE_HANDS = 1 << 1
FEATURE_SIDE_TO_MOVE = 1 << 2
FEATURE_KINGS = 1 << 3
FEATURE_ATTACKS = 1 << 4

_BASE_MAXIMUMS = {"P": 18, "L": 4, "N": 4, "S": 4, "G": 4, "B": 2, "R": 2, "K": 2}
_PROMOTABLE = frozenset({"P", "L", "N", "S", "B", "R"})


@dataclass(frozen=True, slots=True)
class Piece:
    """An absolute-color piece on one canonical board square."""

    side: int
    kind: str


@dataclass(frozen=True, slots=True)
class ParsedSfen:
    """Validated SFEN values used by every feature group."""

    board: tuple[Piece | None, ...]
    hands: tuple[tuple[int, ...], tuple[int, ...]]
    side_to_move: int
    move_number: int


@dataclass(frozen=True, slots=True)
class FeatureGroup:
    """One contiguous region of the flat model input vector."""

    name: str
    offset: int
    dimension: int


def feature_groups(config: FeatureConfig) -> tuple[FeatureGroup, ...]:
    """Return enabled groups in their stable serialization order."""

    specifications = (
        ("board_planes", config.board_planes, 2 * len(PIECE_KINDS) * SQUARES),
        ("hand_counts", config.hand_counts, 2 * len(HAND_PIECES)),
        ("side_to_move", config.side_to_move, 1),
        ("king_coordinates", config.king_coordinates, 4),
        ("pseudo_attacks", config.pseudo_attacks, 2 * SQUARES),
    )
    groups: list[FeatureGroup] = []
    offset = 0
    for name, enabled, dimension in specifications:
        if enabled:
            groups.append(FeatureGroup(name=name, offset=offset, dimension=dimension))
            offset += dimension
    return tuple(groups)


def input_dimension(config: FeatureConfig) -> int:
    groups = feature_groups(config)
    return groups[-1].offset + groups[-1].dimension


def feature_flags(config: FeatureConfig) -> int:
    """Return the OSAVAL01 bitmask for enabled feature groups."""

    return (
        (FEATURE_BOARD if config.board_planes else 0)
        | (FEATURE_HANDS if config.hand_counts else 0)
        | (FEATURE_SIDE_TO_MOVE if config.side_to_move else 0)
        | (FEATURE_KINGS if config.king_coordinates else 0)
        | (FEATURE_ATTACKS if config.pseudo_attacks else 0)
    )


def feature_schema(config: FeatureConfig) -> dict[str, Any]:
    """Produce the complete cross-language feature contract."""

    return {
        "schema": "open_shogi_value_features/v1",
        "featureSchemaVersion": FEATURE_SCHEMA_VERSION,
        "configSha256": config_sha256(config),
        "featureFlags": feature_flags(config),
        "inputDimension": input_dimension(config),
        "groups": [
            {"name": group.name, "offset": group.offset, "dimension": group.dimension}
            for group in feature_groups(config)
        ],
        "coordinateOrder": {
            "squareIndex": "rank_a_to_i_times_9_plus_file_9_to_1",
            "kingCoordinates": "black_file,black_rank,white_file,white_rank",
            "normalization": "one_based_shogi_coordinate_minus_5_divided_by_4",
        },
        "boardPlanes": {
            "sideOrder": list(SIDES),
            "pieceKindOrder": list(PIECE_KINDS),
            "layout": "side,piece_kind,square",
        },
        "handCounts": {
            "sideOrder": list(SIDES),
            "pieceOrder": list(HAND_PIECES),
            "normalizationDivisors": list(HAND_MAXIMUMS),
        },
        "sideToMove": {"black": 1.0, "white": -1.0},
        "pseudoAttacks": {
            "sideOrder": list(SIDES),
            "layout": "side,square",
            "semantics": (
                "geometric control ignoring check and pins; a ray includes and stops at its "
                "first occupied square, including a friendly occupied square; a piece does "
                "not mark its own origin square solely by being there"
            ),
        },
        "outputPerspective": "current_side_to_move",
    }


def parse_canonical_sfen(sfen: str) -> ParsedSfen:
    """Parse the strict canonical SFEN subset emitted by the local Rust engine."""

    if not isinstance(sfen, str) or not sfen or not sfen.isascii() or len(sfen) > 1_024:
        raise ValueError("SFEN must be non-empty ASCII of at most 1024 characters")
    fields = sfen.split(" ")
    if len(fields) != 4 or any(not field for field in fields):
        raise ValueError("SFEN must contain exactly four single-space-separated fields")
    board = _parse_board(fields[0])
    if fields[1] not in {"b", "w"}:
        raise ValueError("SFEN side-to-move must be b or w")
    hands = _parse_hands(fields[2])
    if not fields[3].isdecimal() or fields[3].startswith("0"):
        raise ValueError("SFEN move number must be a canonical positive decimal")
    move_number = int(fields[3])
    if move_number > 2**31 - 1:
        raise ValueError("SFEN move number exceeds the supported bound")
    parsed = ParsedSfen(
        board=board,
        hands=hands,
        side_to_move=0 if fields[1] == "b" else 1,
        move_number=move_number,
    )
    _validate_position_inventory(parsed)
    _validate_pawn_and_dead_piece_constraints(parsed)
    return parsed


def extract_features(sfen: str, config: FeatureConfig) -> list[float]:
    """Encode one canonical SFEN into the exact configured flat vector."""

    parsed = parse_canonical_sfen(sfen)
    values: list[float] = []
    if config.board_planes:
        planes = [0.0] * (2 * len(PIECE_KINDS) * SQUARES)
        kind_indices = {kind: index for index, kind in enumerate(PIECE_KINDS)}
        for square, piece in enumerate(parsed.board):
            if piece is not None:
                index = (piece.side * len(PIECE_KINDS) + kind_indices[piece.kind]) * SQUARES
                planes[index + square] = 1.0
        values.extend(planes)
    if config.hand_counts:
        for side in range(2):
            values.extend(
                count / maximum
                for count, maximum in zip(parsed.hands[side], HAND_MAXIMUMS, strict=True)
            )
    if config.side_to_move:
        values.append(1.0 if parsed.side_to_move == 0 else -1.0)
    if config.king_coordinates:
        for side in range(2):
            king_square = next(
                index
                for index, piece in enumerate(parsed.board)
                if piece == Piece(side=side, kind="K")
            )
            row, column = divmod(king_square, BOARD_SIZE)
            file_number = BOARD_SIZE - column
            rank_number = row + 1
            values.extend(((file_number - 5) / 4.0, (rank_number - 5) / 4.0))
    if config.pseudo_attacks:
        values.extend(_pseudo_attack_maps(parsed.board))
    expected = input_dimension(config)
    if len(values) != expected or any(not math.isfinite(value) for value in values):
        raise RuntimeError("feature encoder violated its dimension or finiteness invariant")
    return values


def zero_feature_group(
    features: list[float], config: FeatureConfig, group_name: str
) -> list[float]:
    """Return a copy with one configured group zeroed for validation ablations."""

    if len(features) != input_dimension(config):
        raise ValueError("feature vector has the wrong dimension")
    group = next((item for item in feature_groups(config) if item.name == group_name), None)
    if group is None:
        raise ValueError(f"feature group is not enabled: {group_name}")
    result = list(features)
    result[group.offset : group.offset + group.dimension] = [0.0] * group.dimension
    return result


def _parse_board(board_text: str) -> tuple[Piece | None, ...]:
    ranks = board_text.split("/")
    if len(ranks) != BOARD_SIZE:
        raise ValueError("SFEN board must contain nine ranks")
    board: list[Piece | None] = []
    for rank in ranks:
        squares: list[Piece | None] = []
        index = 0
        while index < len(rank):
            token = rank[index]
            if token.isdigit():
                if token == "0":
                    raise ValueError("SFEN empty-square run must be between 1 and 9")
                squares.extend([None] * int(token))
                index += 1
                continue
            promoted = token == "+"
            if promoted:
                index += 1
                if index >= len(rank):
                    raise ValueError("SFEN promotion marker lacks a piece")
                token = rank[index]
            base = token.upper()
            if base not in _BASE_MAXIMUMS:
                raise ValueError(f"unsupported SFEN board piece: {token!r}")
            if promoted and base not in _PROMOTABLE:
                raise ValueError(f"piece cannot be promoted in SFEN: {token!r}")
            if not token.isalpha() or (token != base and token != base.lower()):
                raise ValueError(f"invalid SFEN piece token: {token!r}")
            kind = f"+{base}" if promoted else base
            squares.append(Piece(side=0 if token.isupper() else 1, kind=kind))
            index += 1
        if len(squares) != BOARD_SIZE:
            raise ValueError("each SFEN rank must expand to exactly nine squares")
        board.extend(squares)
    return tuple(board)


def _parse_hands(hands_text: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    counts = [{piece: 0 for piece in HAND_PIECES} for _ in range(2)]
    if hands_text != "-":
        index = 0
        while index < len(hands_text):
            count_start = index
            while index < len(hands_text) and hands_text[index].isdigit():
                index += 1
            count_text = hands_text[count_start:index]
            if index >= len(hands_text):
                raise ValueError("SFEN hand count lacks a piece")
            token = hands_text[index]
            index += 1
            base = token.upper()
            if base not in HAND_PIECES or not token.isalpha():
                raise ValueError(f"invalid SFEN hand piece: {token!r}")
            count = int(count_text) if count_text else 1
            if count <= 0 or (count_text and (count == 1 or count_text.startswith("0"))):
                raise ValueError("SFEN hand counts must use canonical positive notation")
            side = 0 if token.isupper() else 1
            if counts[side][base] != 0:
                raise ValueError("SFEN hand piece tokens must not be duplicated")
            counts[side][base] = count
        if _serialize_hands(counts) != hands_text:
            raise ValueError("SFEN hands are not in canonical Black/White RBG... order")
    return (
        tuple(counts[0][piece] for piece in HAND_PIECES),
        tuple(counts[1][piece] for piece in HAND_PIECES),
    )


def _serialize_hands(counts: list[dict[str, int]]) -> str:
    tokens: list[str] = []
    for side in range(2):
        for piece in SFEN_HAND_ORDER:
            count = counts[side][piece]
            if count:
                token = piece if side == 0 else piece.lower()
                tokens.append((str(count) if count > 1 else "") + token)
    return "".join(tokens) or "-"


def _validate_position_inventory(parsed: ParsedSfen) -> None:
    totals = {piece: 0 for piece in _BASE_MAXIMUMS}
    king_counts = [0, 0]
    for piece in parsed.board:
        if piece is None:
            continue
        base = piece.kind.removeprefix("+")
        totals[base] += 1
        if base == "K":
            king_counts[piece.side] += 1
    for side in range(2):
        for piece, count in zip(HAND_PIECES, parsed.hands[side], strict=True):
            if count > _BASE_MAXIMUMS[piece]:
                raise ValueError(f"SFEN hand count exceeds inventory for {piece}")
            totals[piece] += count
    if king_counts != [1, 1]:
        raise ValueError("SFEN board must contain exactly one king for each side")
    for piece, count in totals.items():
        if count > _BASE_MAXIMUMS[piece]:
            raise ValueError(f"SFEN total inventory exceeds the maximum for {piece}")


def _validate_pawn_and_dead_piece_constraints(parsed: ParsedSfen) -> None:
    unpromoted_pawn_files: set[tuple[int, int]] = set()
    for square, piece in enumerate(parsed.board):
        if piece is None:
            continue
        row, column = divmod(square, BOARD_SIZE)
        if piece.kind == "P":
            identity = (piece.side, column)
            if identity in unpromoted_pawn_files:
                raise ValueError("SFEN contains two unpromoted pawns on one side/file")
            unpromoted_pawn_files.add(identity)
        last_rank = 0 if piece.side == 0 else 8
        last_two = {0, 1} if piece.side == 0 else {7, 8}
        if piece.kind in {"P", "L"} and row == last_rank:
            raise ValueError("SFEN contains a dead unpromoted pawn or lance")
        if piece.kind == "N" and row in last_two:
            raise ValueError("SFEN contains a dead unpromoted knight")


def _pseudo_attack_maps(board: tuple[Piece | None, ...]) -> list[float]:
    maps = [[0.0] * SQUARES for _ in range(2)]
    for square, piece in enumerate(board):
        if piece is None:
            continue
        row, column = divmod(square, BOARD_SIZE)
        steps, rays = _movement(piece)
        for row_delta, column_delta in steps:
            target_row = row + row_delta
            target_column = column + column_delta
            if _on_board(target_row, target_column):
                maps[piece.side][target_row * BOARD_SIZE + target_column] = 1.0
        for row_delta, column_delta in rays:
            target_row = row + row_delta
            target_column = column + column_delta
            while _on_board(target_row, target_column):
                target = target_row * BOARD_SIZE + target_column
                maps[piece.side][target] = 1.0
                if board[target] is not None:
                    break
                target_row += row_delta
                target_column += column_delta
    return maps[0] + maps[1]


def _movement(piece: Piece) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    forward = -1 if piece.side == 0 else 1
    gold = (
        (forward, -1),
        (forward, 0),
        (forward, 1),
        (0, -1),
        (0, 1),
        (-forward, 0),
    )
    orthogonal = ((-1, 0), (1, 0), (0, -1), (0, 1))
    diagonal = ((-1, -1), (-1, 1), (1, -1), (1, 1))
    if piece.kind == "P":
        return ((forward, 0),), ()
    if piece.kind == "L":
        return (), ((forward, 0),)
    if piece.kind == "N":
        return ((2 * forward, -1), (2 * forward, 1)), ()
    if piece.kind == "S":
        return (
            (forward, -1),
            (forward, 0),
            (forward, 1),
            (-forward, -1),
            (-forward, 1),
        ), ()
    if piece.kind in {"G", "+P", "+L", "+N", "+S"}:
        return gold, ()
    if piece.kind == "K":
        return orthogonal + diagonal, ()
    if piece.kind == "B":
        return (), diagonal
    if piece.kind == "R":
        return (), orthogonal
    if piece.kind == "+B":
        return orthogonal, diagonal
    if piece.kind == "+R":
        return diagonal, orthogonal
    raise RuntimeError(f"unknown parsed piece kind: {piece.kind}")


def _on_board(row: int, column: int) -> bool:
    return 0 <= row < BOARD_SIZE and 0 <= column < BOARD_SIZE
