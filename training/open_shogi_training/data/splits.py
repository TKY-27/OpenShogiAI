"""Addition-stable, game-only dataset split assignment."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

DatasetSplit = Literal["train", "validation", "test"]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BASIS_POINTS = 10_000


@dataclass(frozen=True, slots=True)
class SplitPolicy:
    """Versioned salted-hash split policy.

    Thresholds use basis points so assignment is independent of collection
    size and existing games never move when new games are added.
    """

    salt: str
    validation_basis_points: int = 1_000
    test_basis_points: int = 1_000
    schema: str = "phase3_game_split/v1"

    def __post_init__(self) -> None:
        salt_bytes = self.salt.encode("utf-8")
        if not salt_bytes or len(salt_bytes) > 1_024:
            raise ValueError("split salt must contain between 1 and 1,024 UTF-8 bytes")
        if "\x00" in self.salt:
            raise ValueError("split salt must not contain NUL")
        for name, value in (
            ("validation_basis_points", self.validation_basis_points),
            ("test_basis_points", self.test_basis_points),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.validation_basis_points + self.test_basis_points >= _BASIS_POINTS:
            raise ValueError("validation and test basis points must leave a non-empty train split")
        if self.schema != "phase3_game_split/v1":
            raise ValueError("unknown split policy schema")

    def as_dict(self) -> dict[str, int | str]:
        """Return the complete stable policy representation."""

        return {
            "schema": self.schema,
            "salt": self.salt,
            "saltSha256": hashlib.sha256(self.salt.encode("utf-8")).hexdigest(),
            "validationBasisPoints": self.validation_basis_points,
            "testBasisPoints": self.test_basis_points,
        }


def assign_game_split(game_sha256: str, policy: SplitPolicy) -> DatasetSplit:
    """Assign one canonical game hash without consulting any other games."""

    if _SHA256_RE.fullmatch(game_sha256) is None:
        raise ValueError("game_sha256 must be 64 lowercase hexadecimal characters")
    digest = hashlib.sha256(
        policy.salt.encode("utf-8") + b"\0" + bytes.fromhex(game_sha256)
    ).digest()
    bucket = int.from_bytes(digest[:8], "big") % _BASIS_POINTS
    if bucket < policy.test_basis_points:
        return "test"
    if bucket < policy.test_basis_points + policy.validation_basis_points:
        return "validation"
    return "train"


def split_assignments(
    game_sha256_values: Iterable[str],
    policy: SplitPolicy,
) -> dict[str, DatasetSplit]:
    """Assign unique games and reject duplicate identities in the input."""

    assignments: dict[str, DatasetSplit] = {}
    for game_sha256 in game_sha256_values:
        if game_sha256 in assignments:
            raise ValueError(f"duplicate canonical game hash: {game_sha256}")
        assignments[game_sha256] = assign_game_split(game_sha256, policy)
    return assignments


def assert_no_game_leakage(rows: Iterable[tuple[str, DatasetSplit]]) -> None:
    """Reject a sequence that assigns one game identity to multiple splits."""

    seen: dict[str, DatasetSplit] = {}
    for game_sha256, split in rows:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"unknown split {split!r}")
        previous = seen.setdefault(game_sha256, split)
        if previous != split:
            raise ValueError(f"game {game_sha256} leaks across {previous!r} and {split!r} splits")
