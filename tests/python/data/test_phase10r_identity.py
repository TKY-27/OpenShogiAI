from __future__ import annotations

import json
from pathlib import Path

import yaml
from open_shogi_training.data.phase10r_identity import (
    canonical_game_hash,
    canonical_position_hash,
    history_id,
    resolve_canonical_collision,
    source_game_id,
    transposition_key,
)

ROOT = Path(__file__).resolve().parents[3]
INITIAL = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
AFTER_76FU = "lnsgkgsnl/1r5b1/ppppppppp/9/9/P8/1PPPPPPPP/1B5R1/LNSGKGSNL w - 2"


def test_replay_identities_are_domain_separated_and_history_sensitive() -> None:
    game = canonical_game_hash(INITIAL, ["7g7f"])
    start_history = history_id(INITIAL, [])
    moved_history = history_id(INITIAL, ["7g7f"])
    position = canonical_position_hash(AFTER_76FU)
    transposition = transposition_key(AFTER_76FU)

    assert len({game, start_history, moved_history, position, transposition}) == 5
    assert canonical_game_hash(INITIAL, ["7g7f"]) == game
    assert start_history != moved_history


def test_source_game_identity_binds_archive_member_path_and_bytes() -> None:
    first = source_game_id("a" * 64, "round1/game.csa", "b" * 64)
    assert first == source_game_id("a" * 64, "round1/game.csa", "b" * 64)
    assert first != source_game_id("a" * 64, "round2/game.csa", "b" * 64)


def _collision_row(split: str, artifact: str, parser: str) -> dict[str, object]:
    return {
        "canonical_position_id": "c" * 64,
        "split": split,
        "artifact_id": artifact,
        "source_game_id": f"game-{artifact}",
        "record_id": f"record-{artifact}",
        "position_index": 3,
        "parser_version": parser,
        "normalization_version": "phase10r-source-preserving/v2",
    }


def test_collision_precedence_preserves_final_holdout_and_is_deterministic() -> None:
    rows = [
        _collision_row("train", "z-source", "parser-a/v1"),
        _collision_row("final_holdout", "b-source", "parser-b/v1"),
        _collision_row("final_holdout", "a-source", "parser-b/v1"),
    ]
    decision = resolve_canonical_collision(rows)

    assert decision["winning_split"] == "final_holdout"
    assert decision["owner"]["artifact_id"] == "a-source"
    assert decision["excluded_occurrence_count"] == 2
    assert decision["protected_holdout_collision"] is True
    assert decision["parser_normalization_collision"] is True


def test_identity_policy_and_replay_schema_protect_holdout() -> None:
    config = yaml.safe_load((ROOT / "configs/phase10r/identity.yaml").read_text())
    proof_schema = json.loads((ROOT / "docs/data/phase10r-replay-proof.schema.json").read_text())
    assert config["schema"] == "open_shogiai_phase10r_identity/v2"
    assert proof_schema["properties"]["complete_legal_replay"]["const"] is True
    assert "history_ids" in proof_schema["required"]
    assert config["collision_resolution"]["protected_holdout_may_move_to_training"] is False
