"""Strict, bounded adaptation of the audited AobaZero CSA envelope.

This module is intentionally not a shogi parser.  It recognizes the small
AobaZero record envelope selected for Phase 3, removes its search annotations,
and leaves position and move validation to the independently implemented Rust
CSA parser.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime

_MOVE_RE = re.compile(r"^[+-][0-9]{4}[A-Z]{2}$")
_PI_RE = re.compile(r"^PI(?:[1-9][1-9][A-Z]{2})*$")
_TIME_RE = re.compile(r"^T(?:0|[1-9][0-9]*)(?:\.[0-9]{1,3})?$")
_METADATA_KEY_RE = re.compile(r"^[A-Z0-9_]+$")
_SOURCE_FILENAME_TIMESTAMP_RE = re.compile(
    r"^(?P<timestamp>[0-9]{8}_[0-9]{6})(?:_[A-Za-z0-9.-]+)?\.txt$"
)
_KNOWN_TERMINALS = frozenset(
    {
        "%TORYO",
        "%CHUDAN",
        "%SENNICHITE",
        "%OUTE_SENNICHITE",
        "%ILLEGAL_MOVE",
        "%+ILLEGAL_ACTION",
        "%-ILLEGAL_ACTION",
        "%TIME_UP",
        "%JISHOGI",
        "%KACHI",
        "%HIKIWAKE",
        "%MAX_MOVES",
        "%MATTA",
        "%TSUMI",
        "%FUZUMI",
        "%ERROR",
    }
)


@dataclass(frozen=True, slots=True)
class AobaZeroLimits:
    """Defensive input limits for one individual record."""

    max_bytes: int = 1_048_576
    max_lines: int = 4_096
    max_line_bytes: int = 4_096
    max_moves: int = 2_048
    max_comments: int = 2_048
    max_comment_bytes: int = 262_144
    max_annotation_bytes: int = 524_288
    max_name_bytes: int = 512
    max_metadata_entries: int = 128
    max_metadata_value_bytes: int = 2_048


_DEFAULT_LIMITS = AobaZeroLimits()


@dataclass(frozen=True, slots=True)
class AdaptedAobaZeroCsa:
    """A normalized CSA envelope plus provenance derived from the source bytes."""

    csa: str
    raw_sha256: str
    move_count: int
    black_name: str | None
    white_name: str | None
    metadata: tuple[tuple[str, str], ...]
    terminal: str
    source_datetime: str | None
    ignored_comment_count: int
    stripped_annotation_count: int


class AobaZeroAdaptationError(ValueError):
    """A stable rejection raised for input outside the selected dialect."""

    def __init__(self, code: str, message: str, *, line: int | None = None) -> None:
        self.code = code
        self.line = line
        location = f" at line {line}" if line is not None else ""
        super().__init__(f"{code}{location}: {message}")


def adapt_aobazero_csa(
    raw: bytes,
    *,
    limits: AobaZeroLimits = _DEFAULT_LIMITS,
) -> AdaptedAobaZeroCsa:
    """Adapt one AobaZero individual-game record to explicit CSA V3.0.

    Accepted extensions are bounded apostrophe comment lines and AobaZero move
    search annotations beginning with ``,v=`` or ``,\'``.  The annotation is
    discarded only when the first seven bytes are an exact CSA move token.
    Multiple-game separators, explicit board dialects, unknown terminal codes,
    unsafe metadata, ambiguous trailing content, and incomplete games are
    rejected rather than guessed.
    """

    _validate_limits(limits)
    if not raw:
        raise AobaZeroAdaptationError("empty", "record is empty")
    if len(raw) > limits.max_bytes:
        raise AobaZeroAdaptationError(
            "oversized",
            f"record exceeds {limits.max_bytes} bytes",
        )
    if raw.startswith(b"\xef\xbb\xbf"):
        raise AobaZeroAdaptationError("unsupported_encoding", "UTF-8 BOM is not accepted")
    if b"\x00" in raw:
        raise AobaZeroAdaptationError("unsafe_text", "NUL byte is not accepted")

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AobaZeroAdaptationError(
            "unsupported_encoding",
            "record must be UTF-8 or ASCII",
        ) from error
    if any(_is_disallowed_record_control(character) for character in text):
        raise AobaZeroAdaptationError(
            "unsafe_text",
            "record contains an unsupported control or Unicode line-separator character",
        )

    lines = text.splitlines(keepends=True)
    if len(lines) > limits.max_lines:
        raise AobaZeroAdaptationError(
            "oversized",
            f"record exceeds {limits.max_lines} lines",
        )
    if not lines:
        raise AobaZeroAdaptationError("empty", "record is empty")

    normalized_lines: list[str] = ["V3.0"]
    metadata: list[tuple[str, str]] = []
    metadata_keys: set[str] = set()
    black_name: str | None = None
    white_name: str | None = None
    saw_version = False
    saw_position = False
    saw_side = False
    saw_move = False
    terminal: str | None = None
    move_count = 0
    comment_count = 0
    comment_bytes = 0
    annotation_count = 0
    annotation_bytes = 0
    last_was_move = False
    terminal_line_index: int | None = None
    source_datetime: str | None = None

    for index, raw_line in enumerate(lines):
        line_number = index + 1
        line = _strip_line_ending(raw_line, line_number)
        line_size = len(line.encode("utf-8"))
        if line_size > limits.max_line_bytes:
            raise AobaZeroAdaptationError(
                "oversized",
                f"line exceeds {limits.max_line_bytes} bytes",
                line=line_number,
            )
        if not line:
            if index == len(lines) - 1:
                continue
            raise AobaZeroAdaptationError(
                "corrupt",
                "blank lines are not allowed inside a record",
                line=line_number,
            )
        if line == "/":
            raise AobaZeroAdaptationError(
                "multiple_games",
                "CSA multiple-game separators are not accepted",
                line=line_number,
            )
        if terminal is not None:
            terminal_line_index = terminal_line_index or index
            if line.startswith("'"):
                source_datetime = _merge_source_datetime(
                    source_datetime,
                    _source_datetime_from_comment(line),
                    line_number,
                )
                comment_count, comment_bytes = _account_comment(
                    line,
                    line_number,
                    comment_count,
                    comment_bytes,
                    limits,
                )
                continue
            raise AobaZeroAdaptationError(
                "trailing_content",
                "only comments may follow the terminal line",
                line=line_number,
            )
        if line.startswith("'"):
            source_datetime = _merge_source_datetime(
                source_datetime,
                _source_datetime_from_comment(line),
                line_number,
            )
            comment_count, comment_bytes = _account_comment(
                line,
                line_number,
                comment_count,
                comment_bytes,
                limits,
            )
            last_was_move = False
            continue
        if line.startswith("V"):
            if saw_version or saw_position or saw_side or saw_move or len(normalized_lines) > 1:
                raise AobaZeroAdaptationError(
                    "ambiguous_version",
                    "version is duplicated or not the first non-comment statement",
                    line=line_number,
                )
            if line != "V3.0":
                raise AobaZeroAdaptationError(
                    "unsupported_version",
                    "only the audited CSA V3.0 dialect is accepted",
                    line=line_number,
                )
            saw_version = True
            last_was_move = False
            continue
        if line.startswith("N+") or line.startswith("N-"):
            if saw_position or saw_side:
                raise AobaZeroAdaptationError(
                    "corrupt",
                    "player name appears after the initial-position section",
                    line=line_number,
                )
            side = line[:2]
            name = line[2:]
            _validate_safe_text(name, "player name", line_number, limits.max_name_bytes)
            if side == "N+":
                if black_name is not None:
                    raise AobaZeroAdaptationError(
                        "ambiguous_name",
                        "Black player name is duplicated",
                        line=line_number,
                    )
                black_name = name
            else:
                if white_name is not None:
                    raise AobaZeroAdaptationError(
                        "ambiguous_name",
                        "White player name is duplicated",
                        line=line_number,
                    )
                white_name = name
            normalized_lines.append(line)
            last_was_move = False
            continue
        if line.startswith("$"):
            if saw_position or saw_side:
                raise AobaZeroAdaptationError(
                    "unsafe_metadata",
                    "metadata appears after the initial-position section",
                    line=line_number,
                )
            key, value = _parse_metadata(line, line_number, limits)
            if len(metadata) >= limits.max_metadata_entries:
                raise AobaZeroAdaptationError(
                    "oversized",
                    f"metadata exceeds {limits.max_metadata_entries} entries",
                    line=line_number,
                )
            if key in metadata_keys:
                raise AobaZeroAdaptationError(
                    "ambiguous_metadata",
                    f"metadata key {key!r} is duplicated",
                    line=line_number,
                )
            metadata_keys.add(key)
            metadata.append((key, value))
            normalized_lines.append(line)
            last_was_move = False
            continue
        if line.startswith("PI"):
            if saw_position or saw_side:
                raise AobaZeroAdaptationError(
                    "ambiguous_position",
                    "initial position is duplicated or out of order",
                    line=line_number,
                )
            if _PI_RE.fullmatch(line) is None:
                raise AobaZeroAdaptationError(
                    "unsupported_position",
                    "only the official PI initial-position dialect is accepted",
                    line=line_number,
                )
            saw_position = True
            normalized_lines.append(line)
            last_was_move = False
            continue
        if line.startswith("P"):
            raise AobaZeroAdaptationError(
                "unsupported_position",
                "AobaZero adaptation accepts PI records, not explicit board rows",
                line=line_number,
            )
        if line in {"+", "-"}:
            if not saw_position or saw_side:
                raise AobaZeroAdaptationError(
                    "ambiguous_side",
                    "side-to-move is missing its PI line or is duplicated",
                    line=line_number,
                )
            saw_side = True
            normalized_lines.append(line)
            last_was_move = False
            continue
        if line.startswith("%"):
            if not saw_side:
                raise AobaZeroAdaptationError(
                    "corrupt",
                    "terminal appears before the side-to-move line",
                    line=line_number,
                )
            if line not in _KNOWN_TERMINALS:
                raise AobaZeroAdaptationError(
                    "unsupported_terminal",
                    "terminal is outside the audited official code set",
                    line=line_number,
                )
            terminal = line
            terminal_line_index = index
            normalized_lines.append(line)
            last_was_move = True
            continue
        if line.startswith("T"):
            if not last_was_move or _TIME_RE.fullmatch(line) is None:
                raise AobaZeroAdaptationError(
                    "corrupt",
                    "time must follow one exact move and use the official numeric form",
                    line=line_number,
                )
            # Time is intentionally not part of the Phase 3 normalized dataset.
            last_was_move = False
            continue
        if line.startswith(("+", "-")):
            if not saw_side:
                raise AobaZeroAdaptationError(
                    "corrupt",
                    "move appears before the side-to-move line",
                    line=line_number,
                )
            move, annotation_size = _parse_annotated_move(line, line_number, limits)
            move_count += 1
            if move_count > limits.max_moves:
                raise AobaZeroAdaptationError(
                    "oversized",
                    f"move count exceeds {limits.max_moves}",
                    line=line_number,
                )
            if annotation_size:
                annotation_count += 1
                annotation_bytes += annotation_size
                if annotation_bytes > limits.max_annotation_bytes:
                    raise AobaZeroAdaptationError(
                        "oversized",
                        f"annotations exceed {limits.max_annotation_bytes} bytes",
                        line=line_number,
                    )
            normalized_lines.append(move)
            saw_move = True
            last_was_move = True
            continue
        raise AobaZeroAdaptationError(
            "unsupported_statement",
            "line is outside the selected AobaZero CSA dialect",
            line=line_number,
        )

    if not saw_position:
        raise AobaZeroAdaptationError("incomplete", "PI initial position is missing")
    if not saw_side:
        raise AobaZeroAdaptationError("incomplete", "side-to-move line is missing")
    if terminal is None:
        raise AobaZeroAdaptationError("incomplete", "terminal line is missing")
    if terminal_line_index is None:
        raise AssertionError("terminal index must be recorded")

    return AdaptedAobaZeroCsa(
        csa="\n".join(normalized_lines) + "\n",
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        move_count=move_count,
        black_name=black_name,
        white_name=white_name,
        metadata=tuple(metadata),
        terminal=terminal,
        source_datetime=source_datetime,
        ignored_comment_count=comment_count,
        stripped_annotation_count=annotation_count,
    )


def metadata_date(metadata: tuple[tuple[str, str], ...]) -> str | None:
    """Return the unmodified, safe record date when one is present."""

    values = dict(metadata)
    return values.get("START_TIME") or values.get("DATE")


def metadata_ratings(metadata: tuple[tuple[str, str], ...]) -> tuple[int | None, int | None]:
    """Extract optional decimal ratings from the selected unambiguous keys."""

    values = dict(metadata)
    return (
        _optional_rating(values.get("BLACK_RATING")),
        _optional_rating(values.get("WHITE_RATING")),
    )


def _validate_limits(limits: AobaZeroLimits) -> None:
    for field_name in limits.__dataclass_fields__:
        value = getattr(limits, field_name)
        if value <= 0:
            raise ValueError(f"{field_name} must be positive")


def _strip_line_ending(raw_line: str, line_number: int) -> str:
    if raw_line.endswith("\r\n"):
        line = raw_line[:-2]
    elif raw_line.endswith("\n"):
        line = raw_line[:-1]
    elif raw_line.endswith("\r"):
        raise AobaZeroAdaptationError(
            "corrupt",
            "bare carriage return is not accepted",
            line=line_number,
        )
    else:
        line = raw_line
    if "\r" in line or "\n" in line:
        raise AobaZeroAdaptationError(
            "corrupt",
            "embedded line ending is not accepted",
            line=line_number,
        )
    return line


def _account_comment(
    line: str,
    line_number: int,
    count: int,
    byte_count: int,
    limits: AobaZeroLimits,
) -> tuple[int, int]:
    count += 1
    byte_count += len(line.encode("utf-8"))
    if count > limits.max_comments or byte_count > limits.max_comment_bytes:
        raise AobaZeroAdaptationError(
            "oversized",
            "comment budget is exceeded",
            line=line_number,
        )
    return count, byte_count


def _validate_safe_text(text: str, description: str, line: int, maximum_bytes: int) -> None:
    if not text:
        raise AobaZeroAdaptationError(
            "unsafe_metadata",
            f"{description} must not be empty",
            line=line,
        )
    if len(text.encode("utf-8")) > maximum_bytes:
        raise AobaZeroAdaptationError(
            "oversized",
            f"{description} exceeds {maximum_bytes} bytes",
            line=line,
        )
    if "," in text:
        raise AobaZeroAdaptationError(
            "unsafe_metadata",
            f"{description} contains the CSA statement delimiter",
            line=line,
        )
    if any(_is_unsafe_character(character) for character in text):
        raise AobaZeroAdaptationError(
            "unsafe_metadata",
            f"{description} contains a control or surrogate character",
            line=line,
        )


def _is_unsafe_character(character: str) -> bool:
    category = unicodedata.category(character)
    return category.startswith("C") or category in {"Zl", "Zp"}


def _is_disallowed_record_control(character: str) -> bool:
    category = unicodedata.category(character)
    return category in {"Zl", "Zp"} or (category == "Cc" and character not in {"\r", "\n", "\t"})


def _parse_metadata(
    line: str,
    line_number: int,
    limits: AobaZeroLimits,
) -> tuple[str, str]:
    body = line[1:]
    if ":" not in body:
        raise AobaZeroAdaptationError(
            "unsafe_metadata",
            "metadata must contain ':'",
            line=line_number,
        )
    key, value = body.split(":", 1)
    if _METADATA_KEY_RE.fullmatch(key) is None:
        raise AobaZeroAdaptationError(
            "unsafe_metadata",
            "metadata key contains characters outside A-Z, digits, and underscore",
            line=line_number,
        )
    _validate_safe_text(value, "metadata value", line_number, limits.max_metadata_value_bytes)
    return key, value


def _parse_annotated_move(
    line: str,
    line_number: int,
    limits: AobaZeroLimits,
) -> tuple[str, int]:
    move = line[:7]
    if _MOVE_RE.fullmatch(move) is None:
        raise AobaZeroAdaptationError(
            "corrupt_move",
            "move must begin with one exact seven-byte CSA token",
            line=line_number,
        )
    if len(line) == 7:
        return move, 0
    suffix = line[7:]
    if not (suffix.startswith(",'") or suffix.startswith(",v=")):
        raise AobaZeroAdaptationError(
            "ambiguous_move",
            "only the audited comma-attached AobaZero search annotation is accepted",
            line=line_number,
        )
    if not suffix.isascii() or any(ord(character) < 0x20 for character in suffix):
        raise AobaZeroAdaptationError(
            "unsafe_annotation",
            "search annotation must contain printable ASCII",
            line=line_number,
        )
    annotation_size = len(suffix)
    if annotation_size > limits.max_annotation_bytes:
        raise AobaZeroAdaptationError(
            "oversized",
            f"one search annotation exceeds {limits.max_annotation_bytes} bytes",
            line=line_number,
        )
    return move, annotation_size


def _optional_rating(value: str | None) -> int | None:
    if value is None:
        return None
    if not value.isascii() or not value.isdecimal():
        raise AobaZeroAdaptationError(
            "corrupt_rating",
            "rating metadata must be an unsigned decimal integer",
        )
    rating = int(value)
    if rating > 100_000:
        raise AobaZeroAdaptationError(
            "corrupt_rating",
            "rating metadata exceeds the defensive limit",
        )
    return rating


def _source_datetime_from_comment(line: str) -> str | None:
    match = _SOURCE_FILENAME_TIMESTAMP_RE.fullmatch(line[1:])
    if match is None:
        return None
    try:
        parsed = datetime.strptime(match.group("timestamp"), "%Y%m%d_%H%M%S")
    except ValueError as error:
        raise AobaZeroAdaptationError(
            "corrupt_datetime",
            "source filename comment contains an invalid calendar timestamp",
        ) from error
    return parsed.isoformat(timespec="seconds")


def _merge_source_datetime(
    current: str | None,
    candidate: str | None,
    line_number: int,
) -> str | None:
    if candidate is None or candidate == current:
        return current or candidate
    if current is not None:
        raise AobaZeroAdaptationError(
            "ambiguous_datetime",
            "conflicting source filename timestamps are present",
            line=line_number,
        )
    return candidate
