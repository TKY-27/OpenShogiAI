import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "docs" / "protocol"


def test_protocol_schemas_are_closed_versioned_json() -> None:
    analysis = json.loads((PROTOCOL / "analysis-protocol.schema.json").read_text())
    time_control = json.loads((PROTOCOL / "time-control.schema.json").read_text())
    resource_budget = json.loads((PROTOCOL / "resource-budget.schema.json").read_text())
    opening = json.loads((PROTOCOL / "opening-book.schema.json").read_text())

    assert analysis["$schema"].endswith("2020-12/schema")
    assert time_control["properties"]["schema"]["const"] == "open_shogi_time_control/v1"
    assert time_control["additionalProperties"] is False
    assert time_control["properties"]["safetyMarginMs"]["maximum"] == 1_000
    assert time_control["properties"]["nodes"]["maximum"] == 1_000_000_000
    assert resource_budget["properties"]["schema"]["const"] == ("open_shogi_resource_budget/v1")
    assert resource_budget["additionalProperties"] is False
    assert resource_budget["properties"]["playThreads"]["const"] == 1
    assert resource_budget["properties"]["analysisThreads"]["maximum"] == 1
    assert analysis["$defs"]["start"]["properties"]["schema"]["const"] == ("open_shogi_analysis/v1")
    assert analysis["$defs"]["start"]["additionalProperties"] is False
    assert analysis["$defs"]["start"]["properties"]["multiPv"]["maximum"] == 10
    assert opening["properties"]["schema"]["const"] == "open_shogi_opening_book/v2"
    assert opening["additionalProperties"] is False
    assert opening["$defs"]["candidate"]["additionalProperties"] is False
    assert opening["$defs"]["candidate"]["properties"]["teacherDepth"]["maximum"] == 64


def test_reference_types_use_the_same_schema_names() -> None:
    types = (PROTOCOL / "analysis-types.ts").read_text()
    documentation = (PROTOCOL / "ANALYSIS_PROTOCOL.md").read_text()
    for schema in (
        "open_shogi_analysis/v1",
        "open_shogi_time_control/v1",
        "open_shogi_resource_budget/v1",
    ):
        assert schema in types
        assert schema in documentation
