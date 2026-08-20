import hashlib

import pytest
from open_shogi_training.data.splits import (
    SplitPolicy,
    assert_no_game_leakage,
    assign_game_split,
    split_assignments,
)


def test_assignments_are_exact_deterministic_and_addition_stable() -> None:
    policy = SplitPolicy(
        salt="open-shogi-phase3-aobazero-v1",
        validation_basis_points=1_500,
        test_basis_points=500,
    )
    hashes = [hashlib.sha256(f"game-{index}".encode()).hexdigest() for index in range(8)]

    initial = split_assignments(hashes[:4], policy)
    expanded = split_assignments(hashes, policy)

    assert {game: expanded[game] for game in hashes[:4]} == initial
    for game_hash in hashes:
        digest = hashlib.sha256(policy.salt.encode() + b"\0" + bytes.fromhex(game_hash)).digest()
        bucket = int.from_bytes(digest[:8], "big") % 10_000
        expected = "test" if bucket < 500 else "validation" if bucket < 2_000 else "train"
        assert assign_game_split(game_hash, policy) == expected
    assert policy.as_dict()["salt"] == "open-shogi-phase3-aobazero-v1"


def test_rejects_duplicate_games_and_game_leakage() -> None:
    policy = SplitPolicy(salt="public-reproducibility-salt")
    game_hash = "a" * 64

    with pytest.raises(ValueError, match="duplicate canonical"):
        split_assignments([game_hash, game_hash], policy)
    with pytest.raises(ValueError, match="leaks"):
        assert_no_game_leakage([(game_hash, "train"), (game_hash, "test")])


@pytest.mark.parametrize(
    "policy",
    [
        lambda: SplitPolicy(salt=""),
        lambda: SplitPolicy(salt="bad\0salt"),
        lambda: SplitPolicy(salt="salt", validation_basis_points=5_000, test_basis_points=5_000),
    ],
)
def test_rejects_invalid_split_policies(policy) -> None:
    with pytest.raises(ValueError):
        policy()
