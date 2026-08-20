from __future__ import annotations

import json
from pathlib import Path

import open_shogi_training.selfplay.common as common_module
import pytest
from open_shogi_training.selfplay.__main__ import main
from open_shogi_training.selfplay.common import (
    ContractError,
    artifact_ref,
    load_json,
    replace_json_state,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_json_loader_uses_an_explicit_artifact_node_budget(tmp_path: Path) -> None:
    artifact = tmp_path / "bounded.json"
    artifact.write_text("[0,0,0]\n", encoding="utf-8")

    with pytest.raises(ContractError, match="exceeds 2 nodes"):
        load_json(artifact, maximum_nodes=2)
    assert load_json(artifact, maximum_nodes=4) == [0, 0, 0]


def test_json_loader_rejects_ancestor_replacement_after_same_fd_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    moved = tmp_path / "moved"
    source.mkdir()
    artifact = source / "bounded.json"
    artifact.write_text('{"source":true}\n', encoding="utf-8")
    real_read_descriptor = common_module._read_descriptor

    def swap_after_read(*args, **kwargs):
        result = real_read_descriptor(*args, **kwargs)
        source.rename(moved)
        source.mkdir()
        (source / artifact.name).write_text('{"source":false}\n', encoding="utf-8")
        return result

    monkeypatch.setattr(common_module, "_read_descriptor", swap_after_read)

    with pytest.raises(ContractError, match="ancestor changed"):
        load_json(artifact)


def test_state_cleanup_preserves_a_foreign_temp_relink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "state.json"
    owned = tmp_path / "owned-temp"
    real_publish = common_module.publish_regular_at

    def relink_target_after_publish(*args, **kwargs):
        real_publish(*args, **kwargs)
        target.rename(owned)
        target.write_bytes(b"foreign-do-not-delete")

    monkeypatch.setattr(common_module, "publish_regular_at", relink_target_after_publish)

    with pytest.raises(ContractError, match="changed during publication"):
        replace_json_state(target, {"schema": "test/v1"})

    assert owned.read_bytes() == b'{"schema":"test/v1"}\n'
    assert target.read_bytes() == b"foreign-do-not-delete"


@pytest.mark.parametrize("swap", ["entry", "ancestor"])
def test_contained_artifact_identity_rejects_path_replacement_after_same_fd_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    swap: str,
) -> None:
    root = tmp_path / "repository"
    source = root / "artifacts"
    source.mkdir(parents=True)
    artifact = source / "evidence.bin"
    artifact.write_bytes(b"trusted evidence")
    moved = root / "moved"
    real_read_descriptor = common_module._read_descriptor

    def swap_after_read(*args, **kwargs):
        result = real_read_descriptor(*args, **kwargs)
        if swap == "entry":
            artifact.rename(source / "original.bin")
            artifact.write_bytes(b"replacement")
        else:
            source.rename(moved)
            source.mkdir()
            (source / artifact.name).write_bytes(b"replacement")
        return result

    monkeypatch.setattr(common_module, "_read_descriptor", swap_after_read)

    with pytest.raises(ContractError, match=r"changed while consumed"):
        artifact_ref(root, "artifacts/evidence.bin")


def test_validate_config_cli_reports_bounded_inputs(
    capsys,
) -> None:
    result = main(
        [
            "--project-root",
            str(PROJECT_ROOT),
            "validate-config",
            "--selfplay-config",
            "configs/selfplay/phase6_smoke.toml",
            "--promotion-policy",
            "configs/generation/phase6_promotion.toml",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 0
    assert payload["games"] == 40
    assert payload["workers"] == 2
    assert captured.err == ""


def test_cli_rejects_path_escape_without_writing(capsys) -> None:
    result = main(
        [
            "--project-root",
            str(PROJECT_ROOT),
            "analyze-arena",
            "--results",
            "../outside.json",
            "--output",
            "artifacts/not-written.json",
        ]
    )

    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    assert "parent" in captured.err
