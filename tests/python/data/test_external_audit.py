import struct

import pytest
from open_shogi_training.data.external_audit import (
    ExternalAuditFormatError,
    measure_exact_overlap,
    measure_split_contamination,
    parse_csa_sample,
    parse_hcpe3_sample,
    parse_hcpe_sample,
    parse_kif_sample,
    parse_packed_sfen_value_sample,
    sample_summary,
)


def _license() -> dict[str, str]:
    return {"decision": "pending", "training_allowed": "pending"}


def test_csa_reader_preserves_move_and_source_annotation() -> None:
    raw = b"V3.0\nPI\n+\n+7776FU,v=0.25,r=0.50\n-3334FU,'v=0.50\n%TORYO\n"

    records = parse_csa_sample(
        raw,
        source_id="test-csa",
        artifact_id="sample.csa",
        license_decision=_license(),
    )

    assert len(records) == 2
    assert records[0]["labels"]["played_move"] == "+7776FU"
    assert records[0]["labels"]["raw_source_score"] == 0.25
    assert records[1]["labels"]["raw_annotation"] == ",'v=0.50"
    assert records[0]["labels"]["wdl"]["raw_terminal"] == "%TORYO"
    assert records[0]["position_identity"]["exact"] is False


def test_kif_reader_preserves_uninterpreted_move_text() -> None:
    raw = (
        "# ----棋譜----\n"
        "手数----指手---------消費時間--\n"
        "   1 ７六歩(77)   ( 0:01/00:00:01)\n"
        "   2 ３四歩(33)\n"
    ).encode()

    records = parse_kif_sample(
        raw,
        source_id="test-kif",
        artifact_id="sample.kif",
        license_decision=_license(),
    )

    assert [record["labels"]["played_move"] for record in records] == ["７六歩(77)", "３四歩(33)"]
    assert records[0]["history_identity"]["namespace"] == "kif_move_prefix"


def test_hcpe_reader_uses_published_fixed_layout() -> None:
    raw = struct.pack("<32shhbB", b"P" * 32, -123, 456, 1, 0)

    records = parse_hcpe_sample(
        raw,
        source_id="test-hcpe",
        artifact_id="sample.hcpe",
        license_decision=_license(),
    )

    assert len(records) == 1
    assert records[0]["labels"]["best_move"] == 456
    assert records[0]["labels"]["raw_source_score"] == -123
    assert records[0]["labels"]["wdl"] == {"raw_code": 1, "normalized": "black_win"}
    assert records[0]["position_identity"]["exact"] is True


def test_hcpe3_reader_handles_variable_candidate_visits() -> None:
    raw = struct.pack("<32sHBB", b"H" * 32, 1, 2, 7)
    raw += struct.pack("<hhH", 123, 88, 2)
    raw += struct.pack("<hH", 123, 3)
    raw += struct.pack("<hH", 456, 1)

    records = parse_hcpe3_sample(
        raw,
        source_id="test-hcpe3",
        artifact_id="sample.hcpe3",
        license_decision=_license(),
    )

    assert len(records) == 1
    assert records[0]["labels"]["best_move"] == 123
    assert records[0]["labels"]["policy_distribution"] == [
        {"move16": 123, "visits": 3, "probability": 0.75},
        {"move16": 456, "visits": 1, "probability": 0.25},
    ]
    assert records[0]["search"]["playouts"] == 4


def test_packed_sfen_reader_supports_binpack_alias_and_overlap() -> None:
    raw = struct.pack("<32shHHbB", b"S" * 32, 100, 321, 24, 0, 0)
    first = parse_packed_sfen_value_sample(
        raw,
        source_id="left",
        artifact_id="left.bin",
        format_name="binpack",
        license_decision=_license(),
    )
    second = parse_packed_sfen_value_sample(
        raw,
        source_id="right",
        artifact_id="right.bin",
        format_name="packed_sfen_value",
        license_decision=_license(),
    )

    overlap = measure_exact_overlap({"left": first, "right": second})
    assert overlap["pairs"][0]["exact_position_overlap_count"] == 1
    assert overlap["pairs"][0]["game_identity_measurable"] is False
    assert sample_summary(first)["formats"] == ["binpack"]


def test_split_contamination_reports_exact_position_overlap() -> None:
    raw = struct.pack("<32shHHbB", b"S" * 32, 100, 321, 24, 0, 0)
    records = parse_packed_sfen_value_sample(
        raw,
        source_id="split-test",
        artifact_id="sample.bin",
        license_decision=_license(),
    )

    report = measure_split_contamination({"train": records, "validation": records})
    assert report["position_overlap_pairs"] == [
        {"left": "train", "right": "validation", "overlap_count": 1, "measured": True}
    ]


@pytest.mark.parametrize(
    ("reader", "raw"),
    [
        (parse_hcpe_sample, b"short"),
        (parse_packed_sfen_value_sample, b"short"),
        (parse_hcpe3_sample, b"short"),
    ],
)
def test_binary_readers_reject_truncated_samples(reader, raw: bytes) -> None:
    with pytest.raises(ExternalAuditFormatError):
        reader(raw, source_id="test", artifact_id="truncated.bin")
