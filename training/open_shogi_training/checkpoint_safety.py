"""Restricted, digest-anchored checkpoint deserialization helpers.

Every consumer of a ``.pt`` file must vouch for its bytes before torch.load
runs. The expected digest always comes from a trust anchor outside the file
being loaded — a reviewed control manifest, a preserved hash receipt, or the
save-time sidecar receipt this module writes — never from the unverified file
itself. The file is read exactly once and deserialization sees exactly the
verified bytes, so a concurrent swap between hashing and loading cannot change
what is unpickled.

``weights_only=True`` with the minimal numpy allowlist below loads every
payload this project writes (model/optimizer/scheduler state, the MT19937
numpy RNG state, python and torch RNG state) on the pinned torch/numpy
versions. It is not a statement that arbitrary untrusted pickles are safe.

Legacy checkpoints written before receipts existed are migrated with
``issue_receipt``: only bytes matching a digest from a trusted summary or
manifest are accepted, the limited deserialization must succeed, and the
original file is never modified.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

#: Real phase10r checkpoints are tens of MB; this only bounds absurd inputs.
MAXIMUM_CHECKPOINT_BYTES = 1 << 30

#: Pickling the MT19937 numpy RNG state requires exactly these four numpy
#: globals (verified by round-trip; dropping any one of them fails), so the
#: weights_only unpickler stays restricted to them.
CHECKPOINT_SAFE_GLOBALS: Final = (
    np._core.multiarray._reconstruct,
    np.ndarray,
    np.dtype,
    np.dtypes.UInt32DType,
)


class CheckpointSafetyError(Exception):
    """Raised for every refusal; callers convert to their own error types."""


def receipt_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.sha256")


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_receipt(
    path: Path,
    *,
    error: Callable[[str], Exception],
    digest: str | None = None,
) -> None:
    """Publish a sidecar digest receipt for the bytes just written to ``path``.

    ``digest`` lets a caller vouch for bytes it already verified in memory;
    when it is given, the receipt describes those bytes even if the file on
    disk changed afterwards (a later load of the changed file then refuses).
    """
    receipt = receipt_path(path)
    temporary = receipt.with_name(f".{receipt.name}.{os.getpid()}.partial")
    try:
        with temporary.open("wb") as handle:
            handle.write(
                f"{digest if digest is not None else _digest_file(path)}\n".encode("ascii")
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, receipt)
    except OSError as caught:
        temporary.unlink(missing_ok=True)
        raise error(f"cannot publish checkpoint digest receipt: {receipt}: {caught}") from caught


def verify_receipt(path: Path, *, error: Callable[[str], Exception]) -> str:
    """Return the declared digest; refuse a missing or unreadable receipt."""
    receipt = receipt_path(path)
    if receipt.is_symlink() or not receipt.is_file():
        raise error(f"checkpoint digest receipt is missing: {receipt}")
    try:
        declared = receipt.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError) as caught:
        raise error(f"checkpoint digest receipt is unreadable: {receipt}") from caught
    return declared


def read_verified_bytes(
    path: Path,
    expected_sha256: str,
    *,
    error: Callable[[str], Exception],
    mismatch_message: str,
) -> bytes:
    """Read the file once and return exactly the bytes the digest vouches for."""
    if path.is_symlink() or not path.is_file():
        raise error(f"checkpoint is not a regular file: {path}")
    try:
        if path.stat().st_size > MAXIMUM_CHECKPOINT_BYTES:
            raise error(f"checkpoint exceeds the size limit: {path}")
        payload = path.read_bytes()
    except OSError as caught:
        raise error(f"checkpoint cannot be read: {path}") from caught
    if len(expected_sha256) != 64 or hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise error(f"{mismatch_message}: {path}")
    return payload


def deserialize(payload: bytes, *, error: Callable[[str], Exception]) -> Any:
    """Deserialize digest-verified bytes under the pinned weights-only allowlist."""
    try:
        with torch.serialization.safe_globals(CHECKPOINT_SAFE_GLOBALS):
            return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except Exception as caught:
        raise error("checkpoint cannot be deserialized with the restricted loader") from caught


def issue_receipt(path: Path, expected_sha256: str) -> Path:
    """Mint a receipt for a legacy checkpoint whose bytes a trusted source vouches for.

    The expected digest must come from a reviewed summary or manifest, never
    from the checkpoint itself. The bytes are verified against it, a limited
    deserialization must succeed, and only then is the sidecar receipt written;
    the original checkpoint bytes stay untouched.
    """
    payload = read_verified_bytes(
        path,
        expected_sha256,
        error=CheckpointSafetyError,
        mismatch_message="checkpoint bytes do not match the expected digest",
    )
    value = deserialize(payload, error=CheckpointSafetyError)
    if not isinstance(value, dict) or not isinstance(value.get("schema"), str):
        raise CheckpointSafetyError(f"checkpoint schema is unidentifiable: {path}")
    # The receipt vouches for exactly the verified, deserialized bytes; if the
    # file changed since the read above, later loads refuse instead of trusting
    # bytes that were never checked.
    write_receipt(
        path,
        error=CheckpointSafetyError,
        digest=hashlib.sha256(payload).hexdigest(),
    )
    return receipt_path(path)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 2:
        print(
            json.dumps(
                {
                    "error": (
                        "usage: python -m open_shogi_training.checkpoint_safety "
                        "CHECKPOINT.pt EXPECTED_SHA256"
                    )
                }
            ),
            flush=True,
        )
        return 2
    path, expected = Path(arguments[0]), arguments[1]
    try:
        receipt = issue_receipt(path, expected)
    except CheckpointSafetyError as caught:
        print(json.dumps({"error": str(caught)}), flush=True)
        return 1
    print(
        json.dumps({"receipt": str(receipt), "sha256": expected}),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
