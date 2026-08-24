from __future__ import annotations

import struct
import zipfile
from pathlib import Path

import pytest
from open_shogi_training.data.phase10r_adapters import (
    normalize_source_bytes,
    write_normalized_jsonl,
)
from open_shogi_training.data.phase10r_archive import (
    Phase10RArchiveError,
    extract_zip,
    inventory_archive,
)
from open_shogi_training.data.phase10r_kif import convert_kif_to_csa
from open_shogi_training.data.phase10r_overlap import (
    deduplicate_records,
    measure_phase10r_overlap,
    split_name,
)
from open_shogi_training.data.phase10r_registry import load_phase10r_registry

REGISTRY = Path("configs/phase10r/source-registry.yaml")


def test_phase10r_registry_is_file_level_and_fail_closed() -> None:
    registry = load_phase10r_registry(REGISTRY)

    assert len(registry.artifacts) >= 60
    assert registry.artifact("wcsc32-kifu").trainable
    assert registry.artifact("wcsc33-kifu").state == "reserved_holdout"
    assert registry.artifact("gct-selfplay-hcpe3").state == "pending_permission"
    assert registry.artifact("bonanza-fv-bin-prior-art").local_only
    assert registry.by_state()["denied"] >= 1


def test_phase10r_normalization_preserves_cp932_kif_annotations(tmp_path: Path) -> None:
    raw = (
        "棋戦\uff1asample\n"
        "手合割\uff1a平手\n"
        "手数----指手---------消費時間--\n"
        " 1 ７六歩(77) ( 0:01/00:00:01)\n"
        "**対局 評価値 42 読み筋 ▲７六歩(77)\n"
        " 2 同　銀(88) ( 0:02/00:00:03)\n"
    ).encode("cp932")

    records = normalize_source_bytes(
        raw,
        format_name="kif",
        source_id="denryu",
        artifact_id="sample",
        source_revision="test-revision",
        license_decision={"state": "approved"},
    )
    assert len(records) == 2
    assert records[0]["source_artifact"]["encoding"] == "cp932"
    assert records[0]["raw_evaluation"]["value"] == 42
    assert records[0]["raw_evaluation"]["pv"] == "▲７六歩(77)"
    assert records[0]["raw_evaluation"]["perspective"] == "unknown"
    assert records[1]["played_move"] == "同 銀(88)"
    assert records[1]["validation"]["status"] == "not_replayed"

    output = tmp_path / "normalized.jsonl.gz"
    report = write_normalized_jsonl(records, output)
    assert report["record_count"] == 2
    assert output.is_file()


def test_phase10r_binary_adapter_retains_raw_score_and_policy() -> None:
    raw = struct.pack("<32shHHbB", b"S" * 32, 100, 321, 24, 0, 0)
    records = normalize_source_bytes(
        raw,
        format_name="packed_sfen_value",
        source_id="nodchip",
        artifact_id="sample",
    )
    assert records[0]["position"]["identity"]["exact"] is True
    assert records[0]["best_move"] == 321
    assert records[0]["raw_evaluation"]["value"] == 100
    assert records[0]["validation"]["status"] == "not_replayed"


def test_phase10r_kif_converter_preserves_promotion_drop_and_terminal() -> None:
    raw = (
        "手合割\uff1a平手\n"
        "先手\uff1ablack\n"
        "後手\uff1awhite\n"
        "手数----指手---------消費時間--\n"
        " 1 ７六歩(77)\n"
        " 2 ３四歩(33)\n"
        " 3 ２二角成(88)\n"
        " 4 同　銀(31)\n"
        " 5 ４五歩打\n"
        " 6 千日手\n"
    ).encode("cp932")
    csa, report = convert_kif_to_csa(raw)
    assert report["encoding"] == "cp932"
    assert report["terminal"] == "%SENNICHITE"
    assert "+7776FU" in csa
    assert "+8822UM" in csa
    assert "-3122GI" in csa
    assert "+0045FU" in csa


def test_phase10r_kif_converter_does_not_invent_terminal() -> None:
    raw = ("手合割\uff1a平手\n手数----指手---------消費時間--\n 1 ７六歩(77)\n").encode("cp932")

    csa, report = convert_kif_to_csa(raw)

    assert report["terminal"] is None
    assert "%CHUDAN" not in csa


def test_phase10r_archive_inventory_and_traversal_rejection(tmp_path: Path) -> None:
    archive_path = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("games/sample.csa", "V2.2\nPI\n+\n%TORYO\n")
    inventory = inventory_archive(archive_path)
    assert inventory.archive_format == "zip"
    assert inventory.entries[0].name == "games/sample.csa"
    output = tmp_path / "extract"
    extract_zip(archive_path, output)
    assert (output / "games" / "sample.csa").is_file()

    malicious = tmp_path / "malicious.zip"
    with zipfile.ZipFile(malicious, "w") as archive:
        archive.writestr("../escape.csa", "bad")
    with pytest.raises(Phase10RArchiveError):
        inventory_archive(malicious)


def test_phase10r_overlap_and_priority_are_deterministic() -> None:
    left = {
        "schema": "phase10r_normalized_record/v1",
        "record_id": "left",
        "position": {"identity": {"namespace": "hcp", "digest_sha256": "a", "exact": True}},
        "history": {"identity": {"namespace": "game", "digest_sha256": "g", "exact": True}},
        "provenance": {"raw_record_sha256": "raw-left"},
    }
    right = dict(left, record_id="right")
    overlap = measure_phase10r_overlap({"a": [left], "b": [right]})
    assert overlap["pairs"][0]["position_overlap_count"] == 1
    kept, report = deduplicate_records({"b": [right], "a": [left]}, source_priority=["a", "b"])
    assert len(kept) == 1
    assert kept[0]["record_id"] == "left"
    assert report["dropped_by_source"]["b"] == 1
    assert split_name(left, salt="phase10r-test") == split_name(left, salt="phase10r-test")
    assert split_name(left, salt="phase10r-test", reserved_holdout=True) == "reserved_holdout"
