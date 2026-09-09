from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from open_shogi_training.phase10r import Phase10RValidationError, verify_frozen_hashes
from open_shogi_training.phase10t_build_compat import BASE_COMMIT, runtime_successor_matches


def test_successor_requires_exact_old_and_new_content(tmp_path: Path, monkeypatch) -> None:
    name = "engine/core/src/lib.rs"
    old = hashlib.sha256(b"old source").hexdigest()
    new = hashlib.sha256(b"new source").hexdigest()
    manifest = tmp_path / "configs/phase10t/runtime-successors.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema": "open_shogiai_phase10t_runtime_successors/v1",
                "base_commit": BASE_COMMIT,
                "files": {name: {"before": old, "after": new}},
            }
        )
    )
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: None)
    monkeypatch.setattr("subprocess.check_output", lambda *args, **kwargs: b"old source")
    assert runtime_successor_matches(tmp_path, name, old, new)
    assert not runtime_successor_matches(tmp_path, name, old, "0" * 64)
    assert not runtime_successor_matches(tmp_path, name, "0" * 64, new)
    monkeypatch.setattr("subprocess.check_output", lambda *args, **kwargs: b"wrong history")
    assert not runtime_successor_matches(tmp_path, name, old, new)


def test_successor_never_overrides_training_or_missing_evidence(tmp_path: Path) -> None:
    assert not runtime_successor_matches(
        tmp_path, "configs/phase10r/dataset-mixture.yaml", "a", "b"
    )
    assert not runtime_successor_matches(tmp_path, "engine/core/src/lib.rs", "a", "b")


def test_historical_campaign_still_rejects_changed_runtime() -> None:
    with pytest.raises(Phase10RValidationError, match="Closed campaign"):
        verify_frozen_hashes(Path(__file__).resolve().parents[2])
