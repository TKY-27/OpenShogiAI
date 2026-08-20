from __future__ import annotations

import base64
import gzip
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
from open_shogi_training.selfplay.common import (
    ArtifactRef,
    ContractError,
    load_json_artifact,
)
from open_shogi_training.selfplay.registry import validate_model_registry

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = PROJECT_ROOT / (
    "tests/python/selfplay/fixtures/registry-chain/registry-chain.json.gz.b64"
)
FIXTURE_SHA256 = "6588f3f309acd63004402b623650291075c4395a505a0af4654a0cec488aea3f"
COMPRESSED_SHA256 = "f49067c4fea8528d229ad08f53706868abeaa040788278c5ba214c52f08814ab"
ENVELOPE_SHA256 = "53a6dc9111aac0005efade3f8000d8741479e8d2f442dfbfcae6f949e7b6d176"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate fixture key: {key}")
        result[key] = value
    return result


def _materialize_registry_fixture(root: Path) -> ArtifactRef:
    fixture = FIXTURE_PATH.read_bytes()
    assert len(fixture) == 105_093
    assert hashlib.sha256(fixture).hexdigest() == FIXTURE_SHA256
    compressed = base64.b64decode(b"".join(fixture.splitlines()), validate=True)
    assert len(compressed) == 77_794
    assert hashlib.sha256(compressed).hexdigest() == COMPRESSED_SHA256
    envelope_bytes = gzip.decompress(compressed)
    assert len(envelope_bytes) == 417_383
    assert hashlib.sha256(envelope_bytes).hexdigest() == ENVELOPE_SHA256
    envelope = json.loads(
        envelope_bytes,
        object_pairs_hook=_unique_object,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    assert isinstance(envelope, dict)
    assert set(envelope) == {
        "schema",
        "expectedDecision",
        "registry",
        "files",
        "blobs",
    }
    assert envelope["schema"] == "phase6_registry_acceptance_fixture/v1"
    assert envelope["expectedDecision"] == "inconclusive"
    blobs = envelope["blobs"]
    files = envelope["files"]
    assert isinstance(blobs, dict) and len(blobs) == 95
    assert isinstance(files, list) and len(files) == 202
    seen_paths: set[str] = set()
    used_blobs: set[str] = set()
    total = 0
    for item in files:
        assert isinstance(item, dict) and set(item) == {"path", "sha256", "size"}
        relative = PurePosixPath(item["path"])
        assert (
            not relative.is_absolute()
            and relative.parts
            and all(part not in {"", ".", ".."} for part in relative.parts)
            and "\\" not in item["path"]
            and "\x00" not in item["path"]
            and item["path"] not in seen_paths
        )
        seen_paths.add(item["path"])
        sha256 = item["sha256"]
        size = item["size"]
        assert isinstance(sha256, str) and _SHA256_RE.fullmatch(sha256)
        assert isinstance(size, int) and not isinstance(size, bool) and 0 <= size <= 8_388_608
        encoded = blobs.get(sha256)
        assert isinstance(encoded, str)
        payload = base64.b64decode(encoded, validate=True)
        assert len(payload) == size
        assert hashlib.sha256(payload).hexdigest() == sha256
        total += size
        assert total <= 16 * 1024 * 1024
        destination = root.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        used_blobs.add(sha256)
    assert used_blobs == set(blobs)
    return ArtifactRef.from_dict(envelope["registry"], "fixture registry")


def test_shared_registry_fixture_validates_the_complete_durable_arena_chain(
    tmp_path: Path,
) -> None:
    registry_ref = _materialize_registry_fixture(tmp_path)
    registry = load_json_artifact(tmp_path, registry_ref)
    validated = validate_model_registry(registry, repository_root=tmp_path)
    assert validated["championModelId"] == "champion-v0"
    assert validated["challengerModelId"] is None
    assert validated["generations"][-1]["status"] == "complete"


def test_shared_registry_fixture_rejects_a_changed_report_even_if_the_registry_is_unchanged(
    tmp_path: Path,
) -> None:
    registry_ref = _materialize_registry_fixture(tmp_path)
    registry = load_json_artifact(tmp_path, registry_ref)
    report = next(tmp_path.glob("artifacts/phase6/**/arena-report.json"))
    original = report.read_bytes()
    changed = original.replace(b'"black_win"', b'"white_win"', 1)
    assert changed != original and len(changed) == len(original)
    report.write_bytes(changed)
    with pytest.raises(ContractError, match="SHA-256 mismatch"):
        validate_model_registry(registry, repository_root=tmp_path)
