from __future__ import annotations

import gzip
from pathlib import Path

from open_shogi_training.phase10s_run import (
    START_POOL_SHA256,
    _arena_command,
    _canonical_sfen,
    _paired_bootstrap_lower,
    _recurrence_facts,
    _sfen_board,
    _start_pool,
)

ROOT = Path(__file__).resolve().parents[2]


def test_start_pool_is_the_frozen_800_position_manifest(tmp_path: Path) -> None:
    target = tmp_path / "artifacts/phase10/start-pool-manifest.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(
        gzip.decompress(
            (ROOT / "tests/fixtures/start-pool/start-pool-manifest.json.gz").read_bytes()
        )
    )
    manifest, positions = _start_pool(tmp_path)
    assert manifest["schema"] == "open_shogiai_phase10_start_pool/v2"
    assert len(positions) == 800
    assert START_POOL_SHA256 == "491a3d54d3d67002fc03fc3644c16b405f5b1dbb662e88e4af260bd45757a1f1"


def test_canonical_sfen_removes_only_move_number() -> None:
    sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
    assert _canonical_sfen(sfen) == " ".join(sfen.split()[:3])


def test_sfen_board_counts_promoted_piece_as_one_square() -> None:
    board = _sfen_board("9/9/9/9/9/9/9/9/4+R4 b - 1")
    assert len(board) == 1
    assert board[(4, 8)] == "+R"


def test_arena_command_adds_cli_move_number_without_changing_pool_sfen() -> None:
    command = _arena_command(
        binary=ROOT / "target/release/open-shogi-cli",
        root=ROOT,
        pair_dir=ROOT / "local/phase10s-runs/test/pair-0000",
        position={"sfen": "9/9/9/9/9/9/9/9/4K4 b -"},
        pair_index=0,
        candidate_model=ROOT / "model.osaval02",
        candidate_model_sha256="a" * 64,
        opponent="handcrafted-experimental",
        opponent_model=None,
        opponent_model_sha256=None,
        git_commit="b" * 40,
        resume=False,
    )
    assert command[command.index("--sfen") + 1].endswith(" b - 1")


def test_recurrence_reports_exact_twofold_and_reversible_cycle() -> None:
    sfen = "9/9/9/9/9/9/9/9/4K4 b - 1"
    moved = "9/9/9/9/9/9/9/4K4/9 b - 1"
    facts = _recurrence_facts([sfen, moved, sfen], ["5i5h", "5h5i"])
    assert facts["repeat_visits"] == 1
    assert facts["twofold_visits"] == 1
    assert facts["threefold_visits"] == 0
    assert facts["reversible_cycles"] == 1


def test_paired_bootstrap_is_deterministic_and_below_perfect_score() -> None:
    first = _paired_bootstrap_lower([1.0, 0.5, 0.0, 1.0], seed=TRAINING_SEED)
    second = _paired_bootstrap_lower([1.0, 0.5, 0.0, 1.0], seed=TRAINING_SEED)
    assert first == second
    assert first is not None
    assert first < 1.0


TRAINING_SEED = 20_260_729
