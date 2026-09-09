from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from open_shogi_training.phase10r import load_canonical_pretraining_mixture
from open_shogi_training.phase10r_execution import (
    CONFIG_PATHS,
    Phase10RExecutionError,
    _configuration_hashes,
    _example_row,
    _legacy_rejection_evidence,
    _outcome_wdl,
    _preparation_relative_path,
    _ReplayFeatureProcess,
    _source_statistics,
    _stream_rows,
    _validate_preparation_configuration_hashes,
    preparation_manifest_path,
)

ROOT = Path(__file__).resolve().parents[2]
MIXTURE = load_canonical_pretraining_mixture(ROOT)
START_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"


def test_outcome_wdl_uses_current_side_to_move_perspective() -> None:
    assert _outcome_wdl("black_win", "black") == 2
    assert _outcome_wdl("black_win", "white") == 0
    assert _outcome_wdl("white_win", "black") == 0
    assert _outcome_wdl("white_win", "white") == 2
    assert _outcome_wdl("draw", "black") == 1
    assert _outcome_wdl("unknown", "white") is None


def test_stream_rows_is_deterministic_and_applies_frozen_weights(tmp_path: Path) -> None:
    paths = {}
    for source in ("aobazero", "wcsc", "denryu"):
        path = tmp_path / f"{source}.jsonl"
        path.write_text(
            f'{{"source":"{source}","raw_targets":{{"source_row":1}}}}\n',
            encoding="utf-8",
        )
        paths[source] = path

    first = list(_stream_rows(paths, 100, MIXTURE))
    second = list(_stream_rows(paths, 100, MIXTURE))

    assert first == second
    assert [row["source"] for row in first].count("aobazero") == 35
    assert [row["source"] for row in first].count("wcsc") == 45
    assert [row["source"] for row in first].count("denryu") == 20
    assert [row["raw_targets"]["stream_index"] for row in first] == list(range(100))
    assert all(row["raw_targets"]["sampling_seed"] == 20_260_729 for row in first)


def test_preparation_rejects_retired_freeze_inventory(tmp_path: Path) -> None:
    # Exercise the original acceptance rule against its exact historical runtime, not
    # the separately authorized Phase 10T successor. No datasets are copied or opened.
    declared = _configuration_hashes(ROOT)
    _validate_preparation_configuration_hashes(ROOT, declared)
    declared["configs/phase10r/frozen-controls.sha256"] = "0" * 64
    with pytest.raises(Phase10RExecutionError, match="configuration inventory is invalid"):
        _validate_preparation_configuration_hashes(ROOT, declared)


def test_preparation_rejects_data_affecting_configuration_drift() -> None:
    declared = _configuration_hashes(ROOT)
    declared[CONFIG_PATHS[0]] = "0" * 64

    with pytest.raises(Phase10RExecutionError, match="configuration hashes are stale"):
        _validate_preparation_configuration_hashes(ROOT, declared)


def test_preparation_rejects_missing_configuration_identity() -> None:
    declared = _configuration_hashes(ROOT)
    declared.pop(CONFIG_PATHS[0])

    with pytest.raises(Phase10RExecutionError, match="configuration inventory is invalid"):
        _validate_preparation_configuration_hashes(ROOT, declared)


def _write_legacy_preparation_fixture(tmp_path: Path) -> tuple[Path, Path]:
    legacy = tmp_path / "phase10r-prepared/1m"
    legacy.mkdir(parents=True)
    manifest_path = legacy / "preparation-manifest.json"
    train_path = legacy / "train.jsonl"
    train_path.write_text("{}\n", encoding="utf-8")
    body = {
        "schema": "open_shogiai_phase10r_preparation/v1",
        "scale": "1m",
        "streamed_examples": 1_000_000,
        "source_stream_counts": {
            "aobazero": 400_000,
            "wcsc": 400_000,
            "denryu": 200_000,
        },
        "files": {
            "train.jsonl": {
                "sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
                "bytes": train_path.stat().st_size,
            }
        },
    }
    body["manifest_sha256"] = hashlib.sha256(
        (json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    return legacy, manifest_path


def test_canonical_preparation_paths_are_scale_scoped() -> None:
    assert preparation_manifest_path(ROOT, "1m").relative_to(ROOT / "local/phase10r-data") == Path(
        "phase10r-prepared/1m-mixture-v2/preparation-manifest.json"
    )
    assert preparation_manifest_path(ROOT, "10m").relative_to(ROOT / "local/phase10r-data") == Path(
        "phase10r-prepared/10m-mixture-v2/preparation-manifest.json"
    )


def test_legacy_rejection_marker_binds_the_canonical_1m_replacement(tmp_path: Path) -> None:
    legacy, manifest_path = _write_legacy_preparation_fixture(tmp_path)
    original = manifest_path.read_bytes()

    first = _legacy_rejection_evidence(tmp_path, MIXTURE)
    second = _legacy_rejection_evidence(tmp_path, MIXTURE)

    assert first == second
    assert first["status"] == "REJECTED_MIXTURE_CONTROL_CONFLICT"
    assert first["marker"] == legacy / "REJECTED_MIXTURE_CONTROL_CONFLICT.json"
    marker = json.loads(first["marker"].read_text(encoding="utf-8"))
    assert marker["legacy_preparation_manifest"] == "phase10r-prepared/1m/preparation-manifest.json"
    assert marker["replacement_preparation_manifest"] == (
        "phase10r-prepared/1m-mixture-v2/preparation-manifest.json"
    )
    assert manifest_path.read_bytes() == original


def test_canonical_mixture_supports_a_valid_10m_replacement() -> None:
    assert MIXTURE["control_id"] == "phase10r-1m-mixture-v2"
    assert _preparation_relative_path("10m", MIXTURE) == Path("phase10r-prepared/10m-mixture-v2")
    assert _preparation_relative_path("1m", MIXTURE) != _preparation_relative_path("10m", MIXTURE)


def test_preparation_rejects_an_incorrect_scale() -> None:
    with pytest.raises(Phase10RExecutionError, match="unsupported preparation scale"):
        _preparation_relative_path("20m", MIXTURE)


def test_preparation_rejects_an_incorrect_mixture_version() -> None:
    mixture = dict(MIXTURE)
    mixture["control_id"] = "phase10r-1m-mixture-v3"

    with pytest.raises(Phase10RExecutionError, match="mixture identity is unsupported"):
        _preparation_relative_path("10m", mixture)


def test_preparation_rejects_incorrect_source_shares() -> None:
    mixture = dict(MIXTURE)
    mixture["target_shares"] = {**MIXTURE["target_shares"], "wcsc": 0.46}

    with pytest.raises(Phase10RExecutionError, match="source shares changed"):
        _preparation_relative_path("10m", mixture)


def test_legacy_rejection_evidence_rejects_an_incorrect_manifest_hash(tmp_path: Path) -> None:
    _, manifest_path = _write_legacy_preparation_fixture(tmp_path)
    body = json.loads(manifest_path.read_text(encoding="utf-8"))
    body["manifest_sha256"] = "0" * 64
    manifest_path.write_text(
        json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )

    with pytest.raises(Phase10RExecutionError, match="manifest digest is invalid"):
        _legacy_rejection_evidence(tmp_path, MIXTURE)


def test_legacy_rejection_evidence_rejects_a_corrupted_marker(tmp_path: Path) -> None:
    _, _ = _write_legacy_preparation_fixture(tmp_path)
    first = _legacy_rejection_evidence(tmp_path, MIXTURE)
    marker_path = first["marker"]
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["replacement_preparation_manifest"] = (
        "phase10r-prepared/10m-mixture-v2/preparation-manifest.json"
    )
    marker_path.write_text(
        json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )

    with pytest.raises(Phase10RExecutionError, match="marker differs from current evidence"):
        _legacy_rejection_evidence(tmp_path, MIXTURE)


def test_preparation_rejects_reusing_the_1m_output_path_for_10m() -> None:
    mixture = dict(MIXTURE)
    mixture["output"] = {"path_template": "phase10r-prepared/1m-mixture-v2"}

    with pytest.raises(Phase10RExecutionError, match="path identity changed"):
        _preparation_relative_path("10m", mixture)


def test_legacy_preparation_is_marked_without_changing_its_manifest(tmp_path: Path) -> None:
    legacy, manifest_path = _write_legacy_preparation_fixture(tmp_path)
    original = manifest_path.read_bytes()

    first = _legacy_rejection_evidence(tmp_path, MIXTURE)
    second = _legacy_rejection_evidence(tmp_path, MIXTURE)

    assert first == second
    assert first["status"] == "REJECTED_MIXTURE_CONTROL_CONFLICT"
    assert manifest_path.read_bytes() == original
    marker = legacy / "REJECTED_MIXTURE_CONTROL_CONFLICT.json"
    assert json.loads(marker.read_text(encoding="utf-8"))["violations"] == [
        {"maximum_share": 0.35, "realized_share": 0.4, "source_id": "aobazero"}
    ]


def test_source_statistics_report_population_and_eligible_repetition() -> None:
    files = {
        "base-train-aobazero.jsonl": {"rows": 7_825},
        "base-train-wcsc.jsonl": {"rows": 356_214},
        "base-train-denryu.jsonl": {"rows": 11_466},
    }
    quotas = {"aobazero": 350_000, "wcsc": 450_000, "denryu": 200_000}
    populations = {"aobazero": 15_488, "wcsc": 618_038, "denryu": 25_993}
    effective = {
        "aobazero": {"all": 14_055, "train": 11_493},
        "wcsc": {"all": 618_038, "train": 490_968},
        "denryu": {"all": 25_993, "train": 20_667},
    }

    result = _source_statistics(files, quotas, 1_000_000, populations, effective, MIXTURE)

    assert result["aobazero"]["maximum_record_occurrences"] == 45
    assert result["wcsc"]["maximum_record_occurrences"] == 2
    assert result["denryu"]["maximum_record_occurrences"] == 18
    assert result["aobazero"]["effective_unique_train_records_before_game_cap"] == 11_493
    assert result["aobazero"]["eligible_train_epoch_equivalent"] == pytest.approx(350_000 / 7_825)


def test_replay_feature_helper_emits_authoritative_legal_root_and_history() -> None:
    with _ReplayFeatureProcess(ROOT) as helper:
        features = helper.game(START_SFEN, ("7g7f", "3c3d"))

    assert len(features) == 3
    assert features[0]["positionIndex"] == 0
    assert "7g7f" in features[0]["legalMoves"]
    assert features[0]["history"] == {
        "available": True,
        "repetitionCount": 1,
        "continuousCheckByUs": False,
        "continuousCheckByThem": False,
    }
    assert features[2]["sfen"].endswith(" b - 3")


def test_replay_feature_helper_rejects_an_illegal_replay() -> None:
    with (
        _ReplayFeatureProcess(ROOT) as helper,
        pytest.raises(RuntimeError, match="illegal USI move"),
    ):
        helper.game(START_SFEN, ("7g7f", "7g7f"))


def test_example_row_keeps_rust_legality_and_source_targets() -> None:
    row = {
        "artifact_id": "aobazero-no-noise-exact100",
        "source_id": "aobazero",
        "source_game_id": "game-1",
        "game_id": "normalized-game-1",
        "record_id": "record-1",
        "position_index": 0,
        "canonical_position_id": "position-1",
        "canonical_sfen": START_SFEN,
        "history_id": "history-1",
        "transposition_key": "transposition-1",
        "side_to_move": "black",
        "split": "train",
        "archive_member_path": "game.csa",
        "archive_member_sha256": "a" * 64,
        "record_sha256": "b" * 64,
        "parser_version": "parser/v1",
        "normalization_version": "normalization/v2",
    }
    payload = {
        "outcome": "black_win",
        "terminal": "TORYO",
        "usi_moves": ["7g7f", "3c3d"],
    }
    with _ReplayFeatureProcess(ROOT) as helper:
        features = helper.game(START_SFEN, payload["usi_moves"])
    result = _example_row(row, payload, features[0])

    assert result["schema"] == "phase10r_training_example/v1"
    assert result["played_move"] == "7g7f"
    assert result["wdl"] == 2
    assert result["wdl_mask"] is True
    assert result["weight"] == 1.0
    assert result["raw_targets"]["legal_mask_runtime"].startswith("open-shogi-core/")


def test_terminal_replay_root_does_not_create_a_policy_target() -> None:
    row = {
        "artifact_id": "aobazero-no-noise-exact100",
        "source_id": "aobazero",
        "source_game_id": "game-1",
        "game_id": "normalized-game-1",
        "record_id": "record-1",
        "position_index": 0,
        "canonical_position_id": "position-1",
        "canonical_sfen": START_SFEN,
        "history_id": "history-1",
        "transposition_key": "transposition-1",
        "side_to_move": "black",
        "split": "train",
        "archive_member_path": "game.csa",
        "archive_member_sha256": "a" * 64,
        "record_sha256": "b" * 64,
        "parser_version": "parser/v1",
        "normalization_version": "normalization/v2",
    }
    feature = {
        "event": "position",
        "positionIndex": 0,
        "sfen": START_SFEN,
        "legalMoves": [],
        "history": {
            "available": True,
            "repetitionCount": 1,
            "continuousCheckByUs": False,
            "continuousCheckByThem": False,
        },
    }
    result = _example_row(
        row,
        {"outcome": "black_win", "terminal": "CHECKMATE", "usi_moves": []},
        feature,
    )

    assert result["legal_moves"] is None
    assert result["played_move"] is None
    assert result["raw_targets"]["terminal_position"] is True
