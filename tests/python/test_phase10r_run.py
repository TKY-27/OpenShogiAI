from __future__ import annotations

import json
from pathlib import Path

import pytest
from open_shogi_training.phase10r_run import (
    BACKEND_GAP,
    Phase10RRunError,
    _backend_receipt,
    _report,
    _write_new_json,
)

ROOT = Path(__file__).resolve().parents[2]


def test_backend_probe_observes_osaval02_parity_runtime() -> None:
    result = _backend_receipt(ROOT)

    assert result["passed"] is True
    assert result["required_format"] == "OSAVAL02"
    assert result["osaval02_parser_present"] is True
    assert result["osaval02_wasm_present"] is True
    assert result["phase10r_model_module_present"] is True
    assert result["stop_reason"] is None


def test_json_receipts_are_immutable(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    _write_new_json(path, {"value": 1})

    with pytest.raises(Phase10RRunError, match="overwrite immutable receipt"):
        _write_new_json(path, {"value": 2})

    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 1}


def test_blocked_report_writes_machine_and_markdown_artifacts(tmp_path: Path) -> None:
    receipt_dir = tmp_path / "local/phase10r-runs"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "preflight.json").write_text(
        json.dumps({"failures": [BACKEND_GAP]}), encoding="utf-8"
    )

    result = _report(tmp_path, ("report", "--scale", "1m"), "1m")

    assert result == 2
    report = json.loads(
        (receipt_dir / "PHASE10R_EXECUTION_REPORT.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "blocked"
    assert BACKEND_GAP in report["stop_reasons"]
    assert (receipt_dir / "PHASE10R_EXECUTION_REPORT.md").is_file()
