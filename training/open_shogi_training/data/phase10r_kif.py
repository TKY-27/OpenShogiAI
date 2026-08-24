"""Bounded KIF-to-CSA conversion for rule-engine validation.

The converter uses the source square printed by KIF and does not infer a move
from ambiguous piece movement.  That keeps the conversion auditable: if a KIF
record omits the source square for a non-drop or uses an unsupported variant,
conversion fails closed and the original KIF remains the only retained data.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_LINE_RE = re.compile(r"^\s*(?P<number>[0-9]+)\s+(?P<first>\S+)(?:\s+(?P<rest>.*))?$")
_SOURCE_RE = re.compile(r"\((?P<source>[0-9]{2})\)$")
_DIGITS = str.maketrans(
    "１２３４５６７８９一二三四五六七八九",
    "123456789123456789",
)
_COORDINATE_RE = re.compile(r"^[1-9][1-9]$")
_PIECES: tuple[tuple[str, str], ...] = (
    ("成香", "NY"),
    ("杏", "NY"),
    ("成桂", "NK"),
    ("圭", "NK"),
    ("成銀", "NG"),
    ("全", "NG"),
    ("と", "TO"),
    ("龍", "RY"),
    ("竜", "RY"),
    ("馬", "UM"),
    ("香", "KY"),
    ("桂", "KE"),
    ("銀", "GI"),
    ("金", "KI"),
    ("角", "KA"),
    ("飛", "HI"),
    ("玉", "OU"),
    ("王", "OU"),
    ("歩", "FU"),
)
_PROMOTIONS = {"FU": "TO", "KY": "NY", "KE": "NK", "GI": "NG", "KA": "UM", "HI": "RY"}
_TERMINALS: tuple[tuple[str, str], ...] = (
    ("投了", "%TORYO"),
    ("詰み", "%TSUMI"),
    ("持将棋", "%JISHOGI"),
    ("中断", "%CHUDAN"),
    ("千日手", "%SENNICHITE"),
    ("時間切れ", "%TIME_UP"),
    ("反則勝ち", "%KACHI"),
    ("入玉宣言", "%JISHOGI"),
    ("入玉勝ち", "%JISHOGI"),
)


class KifConversionError(ValueError):
    """Raised when a KIF move cannot be converted without guessing."""


def convert_kif_to_csa(raw: bytes, *, max_moves: int = 10_000) -> tuple[str, dict[str, Any]]:
    """Convert a standard平手 KIF game to a derived UTF-8 CSA record."""

    text, encoding = _decode(raw)
    headers = _headers(text.splitlines())
    handicap = headers.get("手合割", "平手").strip()
    if handicap and handicap != "平手":
        raise KifConversionError(f"unsupported KIF handicap: {handicap}")
    moves: list[tuple[int, str]] = []
    terminal: str | None = None
    for line in text.splitlines():
        match = _LINE_RE.match(line)
        if match:
            number = int(match.group("number"))
            token = _move_token(match.group("first"), match.group("rest"))
            numbered_terminal = next(
                (csa_terminal for marker, csa_terminal in _TERMINALS if marker in token),
                None,
            )
            if numbered_terminal is not None:
                terminal = numbered_terminal
                continue
            moves.append((number, token))
            if len(moves) > max_moves:
                raise KifConversionError(f"KIF exceeds {max_moves} moves")
            continue
        for marker, csa_terminal in _TERMINALS:
            if marker in line:
                terminal = csa_terminal
                break
    if not moves:
        raise KifConversionError("KIF contains no numbered moves")
    expected = list(range(1, len(moves) + 1))
    observed = [number for number, _ in moves]
    if observed != expected:
        raise KifConversionError(f"KIF move numbers are not contiguous: {observed[:8]}")

    csa_moves: list[str] = []
    previous_destination: str | None = None
    for ply, (_, token) in enumerate(moves):
        move, destination = _convert_move(token, previous_destination)
        csa_moves.append(("+" if ply % 2 == 0 else "-") + move)
        previous_destination = destination
    black = _header_value(headers, "先手")
    white = _header_value(headers, "後手")
    lines = ["'CSA encoding=UTF-8", "V3.0"]
    if black:
        lines.append(f"N+{black}")
    if white:
        lines.append(f"N-{white}")
    lines.extend(["PI", "+", *csa_moves])
    if terminal is not None:
        lines.append(terminal)
    csa = "\n".join(lines) + "\n"
    return csa, {
        "schema": "phase10r_kif_conversion/v2",
        "encoding": encoding,
        "move_count": len(moves),
        "terminal": terminal,
        "source_header_keys": sorted(headers),
    }


def _convert_move(token: str, previous_destination: str | None) -> tuple[str, str]:
    compact = re.sub(r"\s+", "", token).translate(_DIGITS)
    if compact.startswith("同"):
        if previous_destination is None:
            raise KifConversionError("KIF uses 同 on the first move")
        destination = previous_destination
        compact = compact[1:]
    else:
        if len(compact) < 2:
            raise KifConversionError(f"KIF move has no destination: {token!r}")
        destination = compact[:2]
        if not _COORDINATE_RE.fullmatch(destination):
            raise KifConversionError(f"invalid KIF destination: {token!r}")
        compact = compact[2:]
    source_match = _SOURCE_RE.search(compact)
    source = "00" if source_match is None else source_match.group("source")
    if source_match is not None:
        compact = compact[: source_match.start()]
    if source != "00" and not _COORDINATE_RE.fullmatch(source):
        raise KifConversionError(f"invalid KIF source: {token!r}")
    is_drop = "打" in compact
    if is_drop:
        if source != "00":
            raise KifConversionError(f"drop move unexpectedly has a source square: {token!r}")
        compact = compact.replace("打", "")
    piece = next((code for label, code in _PIECES if label in compact), None)
    if piece is None:
        raise KifConversionError(f"unsupported KIF piece notation: {token!r}")
    if "成" in compact and "不成" not in compact:
        piece = _PROMOTIONS.get(piece, piece)
    if source == "00" and not is_drop:
        raise KifConversionError(f"non-drop KIF move has no source square: {token!r}")
    return source + destination + piece, destination


def _headers(lines: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for line in lines:
        if line.startswith("手数"):
            break
        match = re.match(r"^\s*([^:\uFF1A]+)[:\uFF1A](.*)$", line)
        if match:
            headers[match.group(1).strip()] = match.group(2).strip()
    return headers


def _header_value(headers: Mapping[str, str], key: str) -> str | None:
    value = headers.get(key)
    if value is None:
        return None
    return value.replace("\n", " ").replace("\r", " ").strip() or None


def _move_token(first: str, rest: str | None) -> str:
    candidate = " ".join(part for part in (first, rest or "") if part).strip()
    match = re.match(r"^(?P<token>.*?(?:\([0-9]{2}\)|打))(?:\s+\(|$)", candidate)
    return (match.group("token") if match else first).strip()


def _decode(raw: bytes) -> tuple[str, str]:
    for encoding in ("utf-8", "cp932"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise KifConversionError("KIF is neither UTF-8 nor CP932")


__all__ = ["KifConversionError", "convert_kif_to_csa"]
