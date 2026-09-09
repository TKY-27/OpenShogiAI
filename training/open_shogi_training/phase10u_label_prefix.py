"""Read-only round-2 certification and future tail authentication. Never runs a teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

MANIFEST = "configs/phase10u/round2-label-prefix.json"


def canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bound_bytes(root: Path, binding: dict[str, Any]) -> bytes:
    path = (root / binding["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("evidence path escapes repository")
    data = path.read_bytes()
    if digest(data) != binding["sha256"]:
        raise ValueError(f"evidence hash mismatch: {binding['path']}")
    return data


def certify(root: Path, manifest: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    """Return only receipted bytes; never return the unauthenticated labels as training data."""
    if manifest["execution_authorized"] is not False:
        raise ValueError("certification manifest cannot authorize execution")
    data = bound_bytes(root, manifest["original"])
    lines = data.splitlines(keepends=True)
    count = manifest["certified_rows"]
    if type(count) is not int or count != 600 or len(lines) != 686:
        raise ValueError("round-2 600/86 boundary mismatch")
    if any(not line.endswith(b"\n") or not line.strip() for line in lines):
        raise ValueError("incomplete or blank label row")
    prefix, tail = b"".join(lines[:count]), b"".join(lines[count:])
    receipt = json.loads(bound_bytes(root, manifest["stop_receipt"]))
    progress_bytes = bound_bytes(root, manifest["progress"])
    progress = json.loads(progress_bytes)
    if (
        receipt.get("positions_completed") != count
        or receipt.get("labels_path") != manifest["original"]["path"]
        or receipt.get("labels_sha256") != digest(prefix)
        or receipt.get("progress_sha256") != digest(progress_bytes)
        or progress.get("labels_sha256") != digest(prefix)
        or progress.get("positions_completed") != count
        or digest(prefix) != manifest["certified_prefix_sha256"]
        or digest(tail) != manifest["unauthenticated_tail_sha256"]
        or progress.get("teacher") != manifest["teacher"]
    ):
        raise ValueError("receipt/prefix/teacher binding mismatch")
    selection = json.loads(bound_bytes(root, manifest["selection"]))
    selected = sorted(selection["positions_detail"], key=lambda row: row["index"])
    positions = []
    seen = set()
    for row, expected in zip(map(json.loads, lines), selected[:686], strict=True):
        if (
            row.get("split") != "train"
            or row.get("teacher", {}).get("nodes") != 400000
            or row.get("sfen") in seen
            or any(row.get(key) != expected[key] for key in ("index", "sfen", "source"))
        ):
            raise ValueError("label selection/order/train binding mismatch")
        seen.add(row["sfen"])
        positions.append({key: row[key] for key in ("index", "sfen", "source", "split")})
    request = {
        "schema": "open_shogiai_phase10u_tail_reacquisition_request/v1",
        "execution_authorized": False,
        "status": "PREPARED_NOT_EXECUTED",
        "original_sha256": digest(data),
        "certified_prefix_sha256": digest(prefix),
        "unauthenticated_tail_sha256": digest(tail),
        "certified_rows": count,
        "unauthenticated_rows": len(lines) - count,
        "positions": positions[count:],
        "teacher": manifest["teacher"],
        "nodes": 400000,
        "multipv": 3,
        "output_policy": "new_immutable_attempt_only_no_original_mutation",
        "training_authorized": False,
    }
    return prefix, request


def authenticate_tail(root: Path, manifest: dict[str, Any], receipt_path: Path) -> dict[str, Any]:
    """Validate a future independently supplied acquisition receipt; no teacher is launched.

    Like other local receipts this verifies evidence bindings, not external provenance or
    signatures. Newly acquired labels remain subject to the independent training gates.
    """
    _, request = certify(root, manifest)
    receipt = json.loads(receipt_path.read_bytes())
    if (
        receipt.get("schema") != "open_shogiai_phase10u_tail_acquisition_receipt/v1"
        or receipt.get("request_sha256") != digest(canonical(request))
        or receipt.get("teacher") != request["teacher"]
        or receipt.get("nodes") != 400000
        or receipt.get("positions_completed") != 86
        or receipt.get("legality_verified") is not True
    ):
        raise ValueError("future acquisition receipt binding mismatch")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if digest(canonical(unsigned)) != receipt.get("receipt_sha256"):
        raise ValueError("future receipt digest mismatch")
    labels = receipt["labels"]
    if labels["path"] == manifest["original"]["path"]:
        raise ValueError("future acquisition must preserve original evidence")
    lines = bound_bytes(root, labels).splitlines()
    if len(lines) != 86:
        raise ValueError("future acquisition must cover exactly 86 positions")
    for raw, expected in zip(lines, request["positions"], strict=True):
        row = json.loads(raw)
        if (
            any(row.get(key) != value for key, value in expected.items())
            or row.get("teacher", {}).get("nodes") != 400000
            or row.get("teacher", {}).get("name") != "Apery_2.0.0"
        ):
            raise ValueError("future acquisition position/teacher mismatch")
    return {
        "status": "RECEIPT_BOUND_NOT_TRAINING_AUTHORIZED",
        "rows": 86,
        "receipt_sha256": receipt["receipt_sha256"],
        "training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("certify", "prepare-reacquisition", "authenticate"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.root / args.manifest).read_bytes())
    if args.command == "authenticate":
        if args.receipt is None:
            parser.error("authenticate requires --receipt from a future independent acquisition")
        result = authenticate_tail(args.root, manifest, args.receipt)
    else:
        _, result = certify(args.root, manifest)
        if args.command == "certify":
            result = {
                key: result[key]
                for key in (
                    "certified_rows",
                    "unauthenticated_rows",
                    "certified_prefix_sha256",
                    "unauthenticated_tail_sha256",
                    "execution_authorized",
                    "training_authorized",
                )
            }
    print(canonical(result).decode(), end="")


if __name__ == "__main__":
    main()
