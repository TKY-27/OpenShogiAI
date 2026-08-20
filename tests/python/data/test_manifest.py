from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import open_shogi_training.data.manifest as manifest_module
import pytest
from open_shogi_training.data.manifest import (
    CompletedObject,
    EvidenceSnapshot,
    ManifestError,
    ManifestStore,
)
from open_shogi_training.data.registry import LicenseEvidence


def _completed() -> CompletedObject:
    return CompletedObject(
        source_id="source",
        object_id="game",
        url="https://example.test/game.csa",
        retrieved_at="2026-07-29T00:00:00Z",
        sha256="a" * 64,
        size=4,
        content_type="text/plain",
        etag='"v1"',
        last_modified="Wed, 29 Jul 2026 00:00:00 GMT",
        object_path=f"objects/sha256/aa/{'a' * 64}",
        original_filename="game.csa",
        data_format="CSA",
        compression="none",
        license="Public Domain",
        license_evidence=(
            LicenseEvidence(
                url="https://example.test/rights",
                local_path="docs/source-audits/example.md",
                quote="Public domain.",
            ),
        ),
        evidence_snapshots=(
            EvidenceSnapshot(
                evidence_id="rights",
                url="https://example.test/rights",
                retrieved_at="2026-07-29T00:00:00Z",
                sha256="b" * 64,
                size=14,
                content_type="text/plain",
                object_path=f"evidence/sha256/bb/{'b' * 64}",
            ),
        ),
        redistributable=True,
        machine_learning_allowed=True,
    )


def test_manifest_round_trips_completed_records_in_append_order(tmp_path: Path) -> None:
    store = ManifestStore(tmp_path)
    record = _completed()

    store.append_completed(record)

    assert store.completed_records() == (record,)
    raw = store.path.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    assert json.loads(raw)["event"] == "completed"


def test_manifest_append_fsyncs_file_and_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    original = manifest_module.os.fsync

    def recording_fsync(descriptor: int) -> None:
        calls.append(descriptor)
        original(descriptor)

    monkeypatch.setattr(manifest_module.os, "fsync", recording_fsync)

    ManifestStore(tmp_path).append_completed(_completed())

    assert len(calls) >= 2


def test_manifest_rejects_duplicate_json_keys_and_incomplete_tail(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    path.write_text('{"event":"completed","event":"partial"}\n', encoding="utf-8")
    with pytest.raises(ManifestError, match="duplicate key"):
        ManifestStore(tmp_path).snapshot()

    path.write_text('{"event":"partial"}', encoding="utf-8")
    with pytest.raises(ManifestError, match="incomplete final line"):
        ManifestStore(tmp_path).snapshot()


def test_manifest_rejects_whitespace_only_validator() -> None:
    record = _completed()
    invalid = replace(record, etag="   ")

    with pytest.raises(ManifestError, match="etag is invalid"):
        ManifestStore(Path("/unused")).append_completed(invalid)


def test_manifest_refuses_symlink_target(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("", encoding="utf-8")
    (tmp_path / "manifest.jsonl").symlink_to(target)

    with pytest.raises(ManifestError, match="open manifest"):
        ManifestStore(tmp_path).append_completed(_completed())
    assert target.read_text(encoding="utf-8") == ""

    with pytest.raises(ManifestError, match="non-symlink"):
        ManifestStore(tmp_path).snapshot()


def test_manifest_refuses_append_beyond_readable_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(manifest_module, "MAX_MANIFEST_BYTES", 1)

    with pytest.raises(ManifestError, match="would exceed"):
        ManifestStore(tmp_path).append_completed(_completed())
