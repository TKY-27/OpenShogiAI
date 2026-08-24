from __future__ import annotations

import json
from pathlib import Path

from open_shogi_training.data.phase10r_scan import scan_records

SFEN_INITIAL = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
SFEN_AFTER = "lnsgkgsnl/1r5b1/ppppppppp/9/9/7P1/PPPPPPP1P/1B5R1/LNSGKGSNL w - 2"


def _metadata(artifact_id: str) -> dict[str, str]:
    return {
        "source_id": "aobazero",
        "artifact_sha256": "a" * 64,
        "source_revision": "test-revision",
        "parser_version": "test-parser/v1",
        "normalization_version": "test-normalization/v1",
        "input_sha256": ("b" if artifact_id == "a" else "c") * 64,
    }


def _row(
    *,
    artifact_id: str,
    record_id: str,
    game_id: str,
    position_index: int,
    sfen: str,
    split: str,
    history_id: str,
    transposition_key: str,
) -> dict[str, object]:
    return {
        "schema": "open_shogiai_phase10r_scanned_position/v1",
        "artifact_id": artifact_id,
        "source_id": "aobazero",
        "record_id": record_id,
        "record_sha256": record_id,
        "game_id": game_id,
        "position_index": position_index,
        "canonical_sfen": sfen,
        "history_id": history_id,
        "transposition_key": transposition_key,
        "side_to_move": "black" if " b " in sfen else "white",
        "split": split,
    }


def _inputs() -> dict[str, dict[str, str]]:
    return {
        "a": {"status": "ready", "path": "a.jsonl", "input_sha256": "b" * 64},
        "b": {"status": "ready", "path": "b.jsonl", "input_sha256": "c" * 64},
    }


def test_scan_is_disk_backed_resumable_and_proves_clean_identity(tmp_path: Path) -> None:
    rows = {
        "a": [
            _row(
                artifact_id="a",
                record_id="d" * 64,
                game_id="game-a",
                position_index=0,
                sfen=SFEN_INITIAL,
                split="train",
                history_id="history-a",
                transposition_key="transposition-a",
            )
        ],
        "b": [
            _row(
                artifact_id="b",
                record_id="e" * 64,
                game_id="game-b",
                position_index=0,
                sfen=SFEN_AFTER,
                split="validation",
                history_id="history-b",
                transposition_key="transposition-b",
            )
        ],
    }
    result = scan_records(
        rows,
        output_dir=tmp_path / "scan",
        approved_artifact_ids=("a", "b"),
        source_metadata={"a": _metadata("a"), "b": _metadata("b")},
        input_identities=_inputs(),
    )

    assert result["status"] == "passed"
    assert result["position_count"] == 2
    assert result["leakage"]["classes"]["prohibited_transposition_overlap"]["status"] == "passed"
    assert (tmp_path / "scan/phase10r-scan.sqlite3").is_file()
    assert (tmp_path / "scan/phase10r-scan-checkpoint.json").is_file()
    manifest = (tmp_path / "scan/phase10r-leakage-manifest.jsonl").read_text(encoding="utf-8")
    assert len(manifest.splitlines()) == 2

    resumed = scan_records(
        rows,
        output_dir=tmp_path / "scan",
        approved_artifact_ids=("a", "b"),
        source_metadata={"a": _metadata("a"), "b": _metadata("b")},
        input_identities=_inputs(),
    )
    assert resumed["status"] == "passed"
    assert resumed["manifest_sha256"] == result["manifest_sha256"]
    assert (
        len(
            (tmp_path / "scan/phase10r-leakage-manifest.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        == 2
    )


def test_scan_blocks_canonical_cross_split_leakage_and_unavailable_transposition(
    tmp_path: Path,
) -> None:
    row_train = _row(
        artifact_id="a",
        record_id="d" * 64,
        game_id="game-a",
        position_index=0,
        sfen=SFEN_INITIAL,
        split="train",
        history_id="history-a",
        transposition_key=None,
    )
    row_validation = dict(row_train)
    row_validation.update(
        {
            "artifact_id": "b",
            "record_id": "e" * 64,
            "game_id": "game-b",
            "history_id": "history-b",
            "split": "validation",
            "transposition_key": None,
        }
    )
    result = scan_records(
        {"a": [row_train], "b": [row_validation]},
        output_dir=tmp_path / "scan",
        approved_artifact_ids=("a", "b"),
        source_metadata={"a": _metadata("a"), "b": _metadata("b")},
        input_identities=_inputs(),
    )

    assert result["status"] == "blocked"
    assert result["leakage"]["classes"]["canonical_position_cross_split"]["count"] == 1
    assert result["leakage"]["classes"]["canonical_position_cross_split"]["status"] == "failed"
    assert (
        result["leakage"]["classes"]["prohibited_transposition_overlap"]["status"] == "unavailable"
    )
    completion = json.loads(
        (tmp_path / "scan/phase10r-completion-proof.json").read_text(encoding="utf-8")
    )
    assert completion["status"] == "blocked"
