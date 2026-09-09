from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest
import torch
from open_shogi_training.phase10_execution import (
    Phase10ExecutionError,
    Phase10LossWeights,
    adapt_phase10_label,
    canonical_position_sfen,
    history_group_id,
    load_phase10_start_pool_manifest,
    outcome_target,
    phase10_losses,
    start_group_memberships,
    validate_frozen_variant,
)

ROOT = Path(__file__).resolve().parents[2]
START = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"


def _label(*, kind: str = "cp", value: int = 240) -> dict:
    return {
        "position_id": hashlib.sha256(b"position").hexdigest(),
        "canonical_sfen": START,
        "split": "train",
        "stage": "opening",
        "game_id": hashlib.sha256(b"game").hexdigest(),
        "position_index": 0,
        "source_id": "aobazero-no-noise",
        "side_to_move": "black",
        "outcome": "black_win",
        "score": {"kind": kind, "value": value},
        "bestmove": "2g2f",
        "candidates": [
            {
                "multipv": 1,
                "score": {"kind": kind, "value": value},
                "pv": ["2g2f"],
            },
            {
                "multipv": 2,
                "score": {"kind": "cp", "value": value - 20},
                "pv": ["8c8d"],
            },
        ],
    }


def test_canonical_identity_omits_only_move_number() -> None:
    first = canonical_position_sfen(START)
    second = canonical_position_sfen(START.replace(" b - 1", " b - 99"))
    assert first == second
    assert first.endswith(" b -")
    assert history_group_id(START, ["2g2f", "8c8d"]) != history_group_id(START, ["2g2f", "8d8e"])


def test_general_opening_is_an_overlapping_umbrella_control() -> None:
    assert start_group_memberships(START, "2g2f", 0) == ("general_opening", "ibisha")
    assert start_group_memberships(START, "2g2f", 24) == ("ibisha",)
    assert start_group_memberships(START, "", 24) == ("hard_middlegame_endgame",)


def test_outcome_is_converted_to_current_side_perspective() -> None:
    assert outcome_target("black_win", "black") == (1.0, 1.0)
    assert outcome_target("black_win", "white") == (-1.0, 1.0)
    assert outcome_target("unknown", "black") == (0.0, 0.0)


def test_mate_is_masked_from_cp_and_keeps_signed_distance() -> None:
    example = adapt_phase10_label(
        _label(kind="mate", value=-7),
        history_id="h" * 64,
        style_group="hard_middlegame_endgame",
    )
    assert example.cp_mask == 0.0
    assert example.cp_value is None
    assert example.mate_mask == 1.0
    assert example.mate_sign == -1.0
    assert example.mate_distance == 7.0
    assert example.raw_source_value_mask == 0.0


def test_loss_masks_incompatible_target_kinds() -> None:
    value = torch.tensor([0.2, -0.3], requires_grad=True)
    policy = torch.zeros(2, requires_grad=True)
    batch = {
        "cp_target": torch.tensor([0.1, 0.0]),
        "cp_mask": torch.tensor([1.0, 0.0]),
        "wdl_target": torch.tensor([1.0, -1.0]),
        "wdl_mask": torch.tensor([1.0, 1.0]),
        "policy_target": torch.tensor([1.0, 0.0]),
        "policy_mask": torch.tensor([1.0, 1.0]),
        "mate_sign": torch.tensor([0.0, -1.0]),
        "mate_mask": torch.tensor([0.0, 1.0]),
        "ranking_target": torch.tensor([0.1, -1.0]),
        "ranking_mask": torch.tensor([1.0, 0.0]),
    }
    losses = phase10_losses(
        value,
        policy,
        batch,
        Phase10LossWeights(cp=1.0, wdl=0.2, ranking=1.0, mate_margin=1.0, policy=0.05),
    )
    assert losses["ranking_count"].item() == 0.0
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert torch.isfinite(value.grad).all()


@pytest.mark.parametrize(
    ("variant", "input_dim", "training", "exported"),
    [
        ("pure-v1-2x128-control", 2287, 309634, 309505),
        ("pure-v1-3x256-attacks", 2449, 759298, 759041),
    ],
)
def test_frozen_variant_parameter_identity(
    variant: str, input_dim: int, training: int, exported: int
) -> None:
    result = validate_frozen_variant(variant)
    assert result["parameters"] == {"input": input_dim, "training": training, "exported": exported}


def test_unknown_variant_is_rejected() -> None:
    with pytest.raises(Phase10ExecutionError):
        validate_frozen_variant("residual-v0")


def test_frozen_start_pool_has_distinct_reserves_and_zero_holdout_overlap(tmp_path: Path) -> None:
    source = tmp_path / "start-pool.json"
    source.write_bytes(
        gzip.decompress(
            (ROOT / "tests/fixtures/start-pool/start-pool-manifest.json.gz").read_bytes()
        )
    )
    manifest = load_phase10_start_pool_manifest(source)
    positions = manifest["positions"]
    assert len(positions) == 800
    assert len({row["positionId"] for row in positions}) == 800
    assert len({row["canonicalStateSha256"] for row in positions}) == 800
    assert {group: detail["uniqueEligible"] for group, detail in manifest["groups"].items()} == {
        "general_opening": 375,
        "ibisha": 3852,
        "opponent_furibisha": 2097,
        "hard_middlegame_endgame": 3886,
    }
    overlap = json.loads(
        gzip.decompress(
            (ROOT / "tests/fixtures/start-pool/start-pool-overlap-report.json.gz").read_bytes()
        )
    )
    assert overlap["splitLeakage"]["legacyFinalHoldoutCanonicalOverlap"] == 0
    assert overlap["splitLeakage"]["legacyFinalHoldoutHistoryGroupOverlap"] == 0


def test_start_pool_loader_keeps_the_fifty_unique_integrity_gate(tmp_path: Path) -> None:
    source = ROOT / "tests/fixtures/start-pool/start-pool-manifest.json.gz"
    manifest = json.loads(gzip.decompress(source.read_bytes()))
    manifest["groups"]["general_opening"]["uniqueEligible"] = 49
    mutated = tmp_path / "short-start-pool.json"
    mutated.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(Phase10ExecutionError, match="start-pool group is short"):
        load_phase10_start_pool_manifest(mutated)
