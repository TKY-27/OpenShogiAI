import hashlib

import pytest
from open_shogi_training.data.aobazero import (
    AobaZeroAdaptationError,
    AobaZeroLimits,
    adapt_aobazero_csa,
    metadata_ratings,
)


def _record(*lines: str) -> bytes:
    return ("\n".join(lines) + "\n").encode()


def test_adapts_selected_dialect_and_is_idempotent() -> None:
    raw = _record(
        "'20260717_015719_w4745.txt",
        "'ignored bounded comment,with comma\tand tab",
        "N+先手",
        "N-White",
        "$BLACK_RATING:2100",
        "$WHITE_RATING:2050",
        "PI",
        "+",
        "+7776FU,v=0.25,r=0.50,32",
        "T1",
        "-3334FU,'v=0.50,r=0.25,123A",
        "%TORYO",
        "'trailing comment is ignored",
    )

    adapted = adapt_aobazero_csa(raw)

    assert adapted.raw_sha256 == hashlib.sha256(raw).hexdigest()
    assert adapted.source_datetime == "2026-07-17T01:57:19"
    assert adapted.black_name == "先手"
    assert adapted.white_name == "White"
    assert metadata_ratings(adapted.metadata) == (2100, 2050)
    assert adapted.stripped_annotation_count == 2
    assert adapted.ignored_comment_count == 3
    assert adapted.csa == (
        "V3.0\n"
        "N+先手\n"
        "N-White\n"
        "$BLACK_RATING:2100\n"
        "$WHITE_RATING:2050\n"
        "PI\n"
        "+\n"
        "+7776FU\n"
        "-3334FU\n"
        "%TORYO\n"
    )
    assert adapt_aobazero_csa(adapted.csa.encode()).csa == adapted.csa


@pytest.mark.parametrize(
    ("lines", "code"),
    [
        (("V2.2", "PI", "+", "%TORYO"), "unsupported_version"),
        (("P1 *  *  *  *  *  *  *  *  * ", "+", "%TORYO"), "unsupported_position"),
        (("PI", "+", "+7776FU,T1", "%TORYO"), "ambiguous_move"),
        (("PI", "+", "%MADE_UP"), "unsupported_terminal"),
        (("PI", "+", "%TORYO", "+7776FU"), "trailing_content"),
        (("PI", "+", "%TORYO", "/"), "multiple_games"),
        (("$EVENT:unsafe,value", "PI", "+", "%TORYO"), "unsafe_metadata"),
    ],
)
def test_rejects_unselected_or_ambiguous_dialect(
    lines: tuple[str, ...],
    code: str,
) -> None:
    with pytest.raises(AobaZeroAdaptationError) as raised:
        adapt_aobazero_csa(_record(*lines))

    assert raised.value.code == code


def test_rejects_conflicting_exact_filename_timestamps() -> None:
    raw = _record(
        "'20260717_015719_w4745.txt",
        "'20260717_020000_w4745.txt",
        "'2026-07-17 01:59:53 this is not a filename timestamp",
        "PI",
        "+",
        "%TORYO",
    )

    with pytest.raises(AobaZeroAdaptationError) as raised:
        adapt_aobazero_csa(raw)

    assert raised.value.code == "ambiguous_datetime"


def test_enforces_independent_comment_move_and_annotation_caps() -> None:
    comment_limited = AobaZeroLimits(max_comments=1)
    with pytest.raises(AobaZeroAdaptationError, match="comment budget"):
        adapt_aobazero_csa(
            _record("'one", "'two", "PI", "+", "%TORYO"),
            limits=comment_limited,
        )

    move_limited = AobaZeroLimits(max_moves=1)
    with pytest.raises(AobaZeroAdaptationError, match="move count"):
        adapt_aobazero_csa(
            _record("PI", "+", "+7776FU", "-3334FU", "%TORYO"),
            limits=move_limited,
        )

    annotation_limited = AobaZeroLimits(max_annotation_bytes=10)
    with pytest.raises(AobaZeroAdaptationError, match="annotations exceed"):
        adapt_aobazero_csa(
            _record(
                "PI",
                "+",
                "+7776FU,v=1234",
                "-3334FU,v=5678",
                "%TORYO",
            ),
            limits=annotation_limited,
        )


def test_move_limit_accepts_2048_records_and_rejects_only_the_next_move() -> None:
    moves = tuple("+7776FU" if index % 2 == 0 else "-3334FU" for index in range(2_049))

    accepted = adapt_aobazero_csa(_record("PI", "+", *moves[:2_048], "%MAX_MOVES"))
    assert accepted.move_count == 2_048

    with pytest.raises(AobaZeroAdaptationError, match="move count exceeds 2048"):
        adapt_aobazero_csa(_record("PI", "+", *moves, "%MAX_MOVES"))


def test_rejects_non_utf8_and_missing_terminal() -> None:
    with pytest.raises(AobaZeroAdaptationError) as encoding_error:
        adapt_aobazero_csa(b"\xffPI\n+\n%TORYO\n")
    assert encoding_error.value.code == "unsupported_encoding"

    with pytest.raises(AobaZeroAdaptationError) as incomplete_error:
        adapt_aobazero_csa(_record("PI", "+", "+7776FU"))
    assert incomplete_error.value.code == "incomplete"

    with pytest.raises(AobaZeroAdaptationError) as unsafe_error:
        adapt_aobazero_csa(_record("$EVENT:\u202eunsafe", "PI", "+", "%TORYO"))
    assert unsafe_error.value.code == "unsafe_metadata"
