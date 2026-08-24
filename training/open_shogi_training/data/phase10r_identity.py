"""Deterministic Phase 10R replay identities and collision precedence."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any, Final

from open_shogi_training.labeling.schema import canonical_state

IDENTITY_SCHEMA: Final = "open_shogiai_phase10r_identity/v2"
REPLAY_PROOF_SCHEMA: Final = "open_shogiai_phase10r_replay_proof/v2"
SPLIT_PRECEDENCE: Final = (
    "public_test",
    "reserved_holdout",
    "final_holdout",
    "source_held_out",
    "validation",
    "train",
)
_SPLIT_RANK: Final = {split: rank for rank, split in enumerate(SPLIT_PRECEDENCE)}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_USI_MOVE = re.compile(r"(?:[1-9][a-i][1-9][a-i]\+?|[RBGSNLP]\*[1-9][a-i])\Z")


class Phase10RIdentityError(ValueError):
    """Raised when replay identity evidence is incomplete or ambiguous."""


def _safe_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise Phase10RIdentityError(f"{field} must be a non-empty NUL-free string")
    return value


def _sha256(value: object, field: str) -> str:
    text = _safe_text(value, field)
    if _SHA256.fullmatch(text) is None:
        raise Phase10RIdentityError(f"{field} must be a lowercase SHA-256")
    return text


def _canonical_sfen(value: object, field: str) -> str:
    text = _safe_text(value, field)
    parse_input = text if len(text.split(" ")) == 4 else f"{text} 1"
    try:
        return canonical_state(parse_input)
    except (TypeError, ValueError) as error:
        raise Phase10RIdentityError(f"{field} is not a legal canonicalizable SFEN") from error


def _digest(domain: str, fields: Iterable[str]) -> str:
    digest = hashlib.sha256()
    domain_bytes = domain.encode("utf-8")
    digest.update(len(domain_bytes).to_bytes(4, "big"))
    digest.update(domain_bytes)
    for field in fields:
        encoded = field.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def source_game_id(
    official_artifact_sha256: str, archive_member_path: str, archive_member_sha256: str
) -> str:
    """Bind one source record to exact official archive and member bytes."""

    artifact = _sha256(official_artifact_sha256, "official_artifact_sha256")
    member_hash = _sha256(archive_member_sha256, "archive_member_sha256")
    member = _safe_text(archive_member_path, "archive_member_path").replace("\\", "/")
    path = PurePosixPath(member)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise Phase10RIdentityError("archive_member_path must be a safe relative path")
    return _digest(
        "open-shogiai/phase10r/source-game/v2",
        (artifact, "/".join(path.parts), member_hash),
    )


def _legal_moves(moves: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    for move in moves:
        if not isinstance(move, str) or _USI_MOVE.fullmatch(move) is None:
            raise Phase10RIdentityError(f"invalid canonical USI move: {move!r}")
        result.append(move)
    return tuple(result)


def canonical_game_hash(initial_sfen: str, moves: Sequence[str]) -> str:
    """Hash the complete legal play, independent of source and result spelling."""

    initial = _canonical_sfen(initial_sfen, "initial_sfen")
    return _digest(
        "open-shogiai/phase10r/canonical-game/v2",
        (initial, *_legal_moves(moves)),
    )


def canonical_position_hash(canonical_sfen: str) -> str:
    canonical = _canonical_sfen(canonical_sfen, "canonical_sfen")
    return _digest("open-shogiai/phase10r/canonical-position/v2", (canonical,))


def transposition_key(canonical_sfen: str) -> str:
    canonical = _canonical_sfen(canonical_sfen, "canonical_sfen")
    return _digest("open-shogiai/phase10r/transposition/v2", (canonical,))


def history_id(initial_sfen: str, move_prefix: Sequence[str]) -> str:
    """Hash the complete legal prefix ending at the represented position."""

    initial = _canonical_sfen(initial_sfen, "initial_sfen")
    return _digest(
        "open-shogiai/phase10r/history/v2",
        (initial, *_legal_moves(move_prefix)),
    )


def resolve_canonical_collision(occurrences: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Choose one occurrence without ever reassigning a protected record."""

    if len(occurrences) < 2:
        raise Phase10RIdentityError("collision resolution requires at least two occurrences")
    identities = {
        _sha256(row.get("canonical_position_id"), "canonical_position_id")
        for row in occurrences
    }
    if len(identities) != 1:
        raise Phase10RIdentityError("collision group mixes canonical position identities")
    splits = {_safe_text(row.get("split"), "split") for row in occurrences}
    unknown = splits - set(_SPLIT_RANK)
    if unknown:
        raise Phase10RIdentityError(f"collision group has unknown splits: {sorted(unknown)}")
    winning_split = min(splits, key=_SPLIT_RANK.__getitem__)
    eligible = [row for row in occurrences if row.get("split") == winning_split]

    def owner_key(row: Mapping[str, Any]) -> tuple[str, str, str, int]:
        index = row.get("position_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise Phase10RIdentityError("position_index must be a non-negative integer")
        return (
            _safe_text(row.get("artifact_id"), "artifact_id"),
            _safe_text(row.get("source_game_id", row.get("game_id")), "source_game_id"),
            _safe_text(row.get("record_id"), "record_id"),
            index,
        )

    owner = min(eligible, key=owner_key)
    owner_game_field = (
        "source_game_id" if owner.get("source_game_id") is not None else "legacy_game_id"
    )
    versions = sorted(
        {
            (
                _safe_text(row.get("parser_version"), "parser_version"),
                _safe_text(row.get("normalization_version"), "normalization_version"),
            )
            for row in occurrences
        }
    )
    return {
        "canonical_position_id": next(iter(identities)),
        "observed_splits": sorted(splits, key=_SPLIT_RANK.__getitem__),
        "winning_split": winning_split,
        "owner": {
            "artifact_id": owner_key(owner)[0],
            owner_game_field: owner_key(owner)[1],
            "record_id": owner_key(owner)[2],
            "position_index": owner_key(owner)[3],
        },
        "occurrence_count": len(occurrences),
        "excluded_occurrence_count": len(occurrences) - 1,
        "parser_normalization_versions": [
            {"parser_version": parser, "normalization_version": normalization}
            for parser, normalization in versions
        ],
        "parser_normalization_collision": len(versions) > 1,
        "protected_holdout_collision": "final_holdout" in splits,
        "action": "retain_owner_exclude_all_other_occurrences_without_reassignment",
    }


__all__ = [
    "IDENTITY_SCHEMA",
    "REPLAY_PROOF_SCHEMA",
    "SPLIT_PRECEDENCE",
    "Phase10RIdentityError",
    "canonical_game_hash",
    "canonical_position_hash",
    "history_id",
    "resolve_canonical_collision",
    "source_game_id",
    "transposition_key",
]
