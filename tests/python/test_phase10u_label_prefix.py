from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from open_shogi_training.phase10u_label_prefix import (
    authenticate_tail,
    canonical,
    certify,
    digest,
)


@pytest.fixture
def evidence(tmp_path: Path) -> tuple[Path, dict]:
    def save(name: str, data: bytes) -> dict:
        (tmp_path / name).write_bytes(data)
        return {"path": name, "sha256": digest(data)}

    rows = [
        {
            "index": i,
            "sfen": f"position-{i}",
            "source": "test",
            "split": "train",
            "teacher": {"nodes": 400000, "name": "Apery_2.0.0"},
        }
        for i in range(686)
    ]
    lines = [canonical(row) for row in rows]
    prefix = digest(b"".join(lines[:600]))
    teacher = {"name": "Apery", "version": "2.0.0", "binary_sha256": "a" * 64}
    progress = save(
        "progress.json",
        canonical({"positions_completed": 600, "labels_sha256": prefix, "teacher": teacher}),
    )
    manifest = {
        "execution_authorized": False,
        "original": save("labels.jsonl", b"".join(lines)),
        "certified_rows": 600,
        "certified_prefix_sha256": prefix,
        "unauthenticated_tail_sha256": digest(b"".join(lines[600:])),
        "teacher": teacher,
        "progress": progress,
        "selection": save("selection.json", canonical({"positions_detail": rows})),
        "stop_receipt": save(
            "stop.json",
            canonical(
                {
                    "positions_completed": 600,
                    "labels_path": "labels.jsonl",
                    "labels_sha256": prefix,
                    "progress_sha256": progress["sha256"],
                }
            ),
        ),
    }
    return tmp_path, manifest


def test_exact_prefix_and_deterministic_request_preserve_original(evidence: tuple) -> None:
    root, manifest = evidence
    original = (root / "labels.jsonl").read_bytes()
    prefix, request = certify(root, manifest)
    assert len(prefix.splitlines()) == 600
    assert request["unauthenticated_rows"] == len(request["positions"]) == 86
    assert request["positions"][0]["index"] == 600
    assert request["positions"][-1]["index"] == 685
    assert request["training_authorized"] is False
    assert canonical(request) == canonical(certify(root, manifest)[1])
    assert (root / "labels.jsonl").read_bytes() == original


@pytest.mark.parametrize("count", [599, 601, 686, True])
def test_no_tail_or_off_by_one_certification(evidence: tuple, count: int) -> None:
    root, manifest = evidence
    manifest["certified_rows"] = count
    with pytest.raises(ValueError, match="boundary"):
        certify(root, manifest)


@pytest.mark.parametrize("name", ["labels.jsonl", "stop.json", "progress.json", "selection.json"])
def test_tampered_evidence_fails_closed(evidence: tuple, name: str) -> None:
    root, manifest = evidence
    with (root / name).open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        certify(root, manifest)


def test_changed_teacher_cannot_certify(evidence: tuple) -> None:
    root, manifest = evidence
    manifest["teacher"] = {"name": "other"}
    with pytest.raises(ValueError, match="binding"):
        certify(root, manifest)


def test_future_authentication_binds_request_and_preserves_training_gate(evidence: tuple) -> None:
    root, manifest = evidence
    _, request = certify(root, manifest)
    data = b"".join((root / "labels.jsonl").read_bytes().splitlines(keepends=True)[600:])
    (root / "new-labels.jsonl").write_bytes(data)
    unsigned = {
        "schema": "open_shogiai_phase10u_tail_acquisition_receipt/v1",
        "request_sha256": digest(canonical(request)),
        "teacher": manifest["teacher"],
        "nodes": 400000,
        "positions_completed": 86,
        "legality_verified": True,
        "labels": {"path": "new-labels.jsonl", "sha256": digest(data)},
    }
    receipt = {**unsigned, "receipt_sha256": digest(canonical(unsigned))}
    path = root / "new-receipt.json"
    path.write_bytes(canonical(receipt))
    assert authenticate_tail(root, manifest, path)["training_authorized"] is False
    for key in ("request_sha256", "teacher", "receipt_sha256", "nodes"):
        tampered = copy.deepcopy(receipt)
        tampered[key] = None
        path.write_text(json.dumps(tampered))
        with pytest.raises(ValueError):
            authenticate_tail(root, manifest, path)


def test_tail_reordering_with_rehashed_source_still_fails_selection(evidence: tuple) -> None:
    root, manifest = evidence
    lines = (root / "labels.jsonl").read_bytes().splitlines(keepends=True)
    lines[600], lines[601] = lines[601], lines[600]
    changed = b"".join(lines)
    (root / "labels.jsonl").write_bytes(changed)
    manifest["original"]["sha256"] = digest(changed)
    manifest["unauthenticated_tail_sha256"] = digest(b"".join(lines[600:]))
    with pytest.raises(ValueError, match="selection"):
        certify(root, manifest)


def test_receipt_cannot_expand_prefix_even_with_rehashed_receipt(evidence: tuple) -> None:
    root, manifest = evidence
    receipt = json.loads((root / "stop.json").read_bytes())
    receipt["positions_completed"] = 686
    raw = canonical(receipt)
    (root / "stop.json").write_bytes(raw)
    manifest["stop_receipt"]["sha256"] = digest(raw)
    with pytest.raises(ValueError, match="binding"):
        certify(root, manifest)
