from __future__ import annotations

import hashlib
import io
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from open_shogi_training.checkpoint_safety import (
    CheckpointSafetyError,
    deserialize,
    issue_receipt,
    read_verified_bytes,
    receipt_path,
    verify_receipt,
    write_receipt,
)


def _payload() -> dict[str, object]:
    return {
        "schema": "open_shogiai_checkpoint_safety_test/v1",
        "model_state": {"w": torch.zeros(4, 4)},
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
    }


def _write_legacy(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_payload(), path)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_receipt_write_verify_and_digest_round_trip(tmp_path: Path) -> None:
    checkpoint = tmp_path / "last.pt"
    _write_legacy(checkpoint)
    write_receipt(checkpoint, error=CheckpointSafetyError)
    receipt = receipt_path(checkpoint)
    assert receipt.is_file()
    assert receipt.read_text(encoding="ascii") == f"{_digest(checkpoint)}\n"
    assert verify_receipt(checkpoint, error=CheckpointSafetyError) == _digest(checkpoint)


def test_deserialize_sees_exactly_the_receipt_verified_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "last.pt"
    _write_legacy(checkpoint)
    write_receipt(checkpoint, error=CheckpointSafetyError)
    declared = verify_receipt(checkpoint, error=CheckpointSafetyError)

    seen: list[str] = []
    real_load = torch.load

    def spy_load(buffer, *args: object, **kwargs: object) -> object:
        buffer.seek(0)
        seen.append(hashlib.sha256(buffer.read()).hexdigest())
        buffer.seek(0)
        return real_load(buffer, *args, **kwargs)

    monkeypatch.setattr(torch, "load", spy_load)
    payload = deserialize(checkpoint.read_bytes(), error=CheckpointSafetyError)
    monkeypatch.undo()
    assert seen == [declared]
    assert payload["schema"] == "open_shogiai_checkpoint_safety_test/v1"


def test_read_verified_bytes_refuses_size_and_hash_drift(tmp_path: Path) -> None:
    checkpoint = tmp_path / "last.pt"
    _write_legacy(checkpoint)
    with pytest.raises(CheckpointSafetyError, match="not match the expected digest"):
        read_verified_bytes(
            checkpoint,
            "0" * 64,
            error=CheckpointSafetyError,
            mismatch_message="checkpoint bytes do not match the expected digest",
        )
    matched = read_verified_bytes(
        checkpoint,
        _digest(checkpoint),
        error=CheckpointSafetyError,
        mismatch_message="checkpoint bytes do not match the expected digest",
    )
    assert hashlib.sha256(matched).hexdigest() == _digest(checkpoint)


def test_issue_receipt_migrates_a_legacy_checkpoint_from_a_trusted_digest(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "first" / "last.pt"
    _write_legacy(checkpoint)
    original = checkpoint.read_bytes()
    trusted = _digest(checkpoint)

    receipt = issue_receipt(checkpoint, trusted)
    assert receipt == receipt_path(checkpoint)
    assert receipt.read_text(encoding="ascii") == f"{trusted}\n"
    # The original checkpoint bytes stay untouched.
    assert checkpoint.read_bytes() == original


def test_issue_receipt_refuses_an_untrusted_or_unreadable_file(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "last.pt"
    _write_legacy(checkpoint)
    with pytest.raises(CheckpointSafetyError, match="not match the expected digest"):
        issue_receipt(checkpoint, "0" * 64)
    assert not receipt_path(checkpoint).exists()

    # A self-consistent digest of arbitrary bytes is not trust: the limited
    # deserialization must refuse it before any receipt exists.
    arbitrary = tmp_path / "arbitrary.pt"
    arbitrary.write_bytes(b"not a checkpoint at all")
    with pytest.raises(CheckpointSafetyError, match="cannot be deserialized"):
        issue_receipt(arbitrary, _digest(arbitrary))
    assert not receipt_path(arbitrary).exists()


def test_command_line_entry_reports_refusals_and_success(tmp_path: Path) -> None:

    checkpoint = tmp_path / "last.pt"
    _write_legacy(checkpoint)
    digest = _digest(checkpoint)

    missing_argument = json.loads(_capture_main([str(checkpoint)]))
    assert "usage" in missing_argument["error"]
    refusal = json.loads(_capture_main([str(checkpoint), "0" * 64]))
    assert "error" in refusal
    assert not receipt_path(checkpoint).exists()

    result = json.loads(_capture_main([str(checkpoint), digest]))
    assert result["sha256"] == digest
    assert receipt_path(checkpoint).is_file()


def _capture_main(arguments: list[str]) -> str:
    import contextlib

    from open_shogi_training.checkpoint_safety import main

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = main(arguments)
    assert code in (0, 1, 2)
    return buffer.getvalue().strip()


def test_read_verified_bytes_refuses_oversized_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from open_shogi_training import checkpoint_safety

    checkpoint = tmp_path / "last.pt"
    _write_legacy(checkpoint)
    monkeypatch.setattr(checkpoint_safety, "MAXIMUM_CHECKPOINT_BYTES", 8)
    with pytest.raises(CheckpointSafetyError, match="size limit"):
        read_verified_bytes(
            checkpoint,
            _digest(checkpoint),
            error=CheckpointSafetyError,
            mismatch_message="checkpoint bytes do not match the expected digest",
        )
