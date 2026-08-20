from __future__ import annotations

import json
from pathlib import Path

from open_shogi_training.evaluation.__main__ import main

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_validate_config_command_is_machine_readable(capsys) -> None:  # type: ignore[no-untyped-def]
    result = main(["--project-root", str(PROJECT_ROOT), "validate-config"])

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "phase7_config_validation/v1"
    assert payload["games"] == 2
    assert payload["autoTrainingEligible"] is False


def test_cli_returns_two_for_invalid_config(capsys, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "bad.toml"
    path.write_text("schema = 'bad'\n")

    result = main(
        [
            "--project-root",
            str(PROJECT_ROOT),
            "--config",
            path.as_posix(),
            "validate-config",
        ]
    )

    assert result == 2
    assert "error:" in capsys.readouterr().err
