"""Bounded, attributable execution for the frozen Phase 10T campaign.

The frozen :mod:`phase10t` module remains the policy validator.  This module owns the actual
execution boundary: it pins the repository, approved train-only inputs, teacher identity,
model identity, runtime receipts, and the fail-closed transition between stages.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from open_shogi_training import phase10t
from open_shogi_training.labeling.config import load_teacher_config, resolve_config_path
from open_shogi_training.labeling.legality import RustLegalityValidator
from open_shogi_training.labeling.usi import USICandidate, USIEngine, USIScore, USISearchResult
from open_shogi_training.phase10t_model import (
    DEFAULT_SEED,
    HistoryFacts,
    Phase10TExample,
    predict_examples,
    train_random_lineage,
    train_supervised_lineage,
)
from open_shogi_training.phase10u_labels import observed_targets, score_order

RECEIPT_SCHEMA: Final = "open_shogiai_phase10t_execution_receipt/v1"
LABEL_SCHEMA: Final = "open_shogiai_phase10t_teacher_label/v1"
RUN_ROOT: Final = Path("local/phase10t-runs")
LOCK_NAME: Final = ".phase10t-run.lock"
MIN_FREE_BYTES: Final = 150 * 1024**3
FROZEN_BRANCH: Final = "codex/pure-learned-pre-selfplay"
FROZEN_ANCESTOR: Final = phase10t.ANCESTOR
TRAIN_SOURCES: Final = (
    Path("local/phase10r-data/phase10r-prepared/1m/base-train-aobazero.jsonl"),
    Path("local/phase10r-data/phase10r-prepared/1m/base-train-wcsc.jsonl"),
    Path("local/phase10r-data/phase10r-prepared/1m/base-train-denryu.jsonl"),
)
APPROVED_SOURCES: Final = frozenset({"aobazero", "wcsc", "denryu"})
RUNG_LIMITS: Final = {"micro": 32, "100k": 100_000, "1m": 1_000_000}
MULTIPV_VALIDATION_RETRIES: Final = 3
FORBIDDEN_PARTS: Final = frozenset(
    {
        "final_holdout",
        "reserved_holdout",
        "public_test",
        "pending_permission",
        "denied",
    }
)


class Phase10TRunError(RuntimeError):
    """Raised when an execution stage cannot prove its frozen preconditions."""


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise Phase10TRunError(f"expected a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_path(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise Phase10TRunError(f"unsafe Phase 10T path: {relative}")
    if any(part.casefold() in FORBIDDEN_PARTS for part in relative.parts):
        raise Phase10TRunError(f"protected holdout/test path is outside this runner: {relative}")
    path = root / relative
    if any(
        (root / Path(*relative.parts[:index])).is_symlink()
        for index in range(1, len(relative.parts) + 1)
    ):
        raise Phase10TRunError(f"symlink path is outside the runner boundary: {relative}")
    try:
        path.resolve(strict=False).relative_to(root.resolve())
    except ValueError as error:
        raise Phase10TRunError(f"path escapes the repository: {relative}") from error
    return path


def _git(root: Path, *arguments: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=check,
        capture_output=True,
        text=True,
    )
    if not check and completed.returncode:
        return completed.stdout.strip()
    return completed.stdout.strip()


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _implementation_lock(root: Path) -> dict[str, Any]:
    raise Phase10TRunError("Closed campaign: historical implementation lock is retired")


def _teacher_identity(root: Path) -> dict[str, Any]:
    config_path = _safe_path(root, Path("configs/teacher/apery-v2.0.0.yaml"))
    config = load_teacher_config(config_path)
    executable = resolve_config_path(root, config.executable, field="teacher.executable")
    eval_files = []
    for item in config.eval_files:
        path = resolve_config_path(root, item.path, field="teacher.eval_file")
        actual = _sha256_file(path)
        if actual != item.sha256:
            raise Phase10TRunError(f"teacher eval hash mismatch: {item.path}")
        eval_files.append({"path": item.path, "sha256": actual, "size": path.stat().st_size})
    binary_sha256 = _sha256_file(executable)
    if binary_sha256 != config.runtime_binary_sha256:
        raise Phase10TRunError("teacher executable hash mismatch")
    return {
        "config_path": config_path.relative_to(root).as_posix(),
        "config_sha256": _sha256_file(config_path),
        "semantic_sha256": config.sha256,
        "name": config.name,
        "version": config.version,
        "binary": {
            "path": config.executable,
            "sha256": binary_sha256,
            "size": executable.stat().st_size,
        },
        "eval_files": eval_files,
        "multipv": config.multipv,
        "configured_nodes": config.nodes,
        "threads": config.threads,
        "hash_mb": config.hash_mb,
        "book_enabled": False,
    }


def _git_identity(root: Path) -> dict[str, Any]:
    branch = _git(root, "branch", "--show-current")
    commit = _git(root, "rev-parse", "HEAD")
    dirty = _git(root, "status", "--porcelain")
    if branch != FROZEN_BRANCH:
        raise Phase10TRunError(f"wrong campaign branch: {branch}")
    if dirty:
        raise Phase10TRunError(
            "worktree is not clean; implementation must be committed before execution"
        )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", FROZEN_ANCESTOR, "HEAD"], cwd=root, check=True
    )
    return {"branch": branch, "commit": commit, "clean": True, "ancestor": FROZEN_ANCESTOR}


def _resource_identity(root: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(root)
    free = usage.free
    if free < MIN_FREE_BYTES:
        raise Phase10TRunError(f"free disk space {free} is below the frozen 150 GiB minimum")
    return {
        "free_bytes": free,
        "minimum_free_bytes": MIN_FREE_BYTES,
        "filesystem_total": usage.total,
    }


def _base_identity(root: Path) -> dict[str, Any]:
    validation = phase10t.validate(root, local=True)
    return {
        "schema": RECEIPT_SCHEMA,
        "created_at": _now(),
        "root": str(root),
        "git": _git_identity(root),
        "resources": _resource_identity(root),
        "frozen_validation": validation,
        "implementation_lock": _implementation_lock(root),
        "teacher": _teacher_identity(root),
    }


@contextlib.contextmanager
def _exclusive_run(root: Path) -> Iterator[None]:
    run_root = _safe_path(root, RUN_ROOT)
    run_root.mkdir(parents=True, exist_ok=True)
    lock_path = run_root / LOCK_NAME
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise Phase10TRunError("another Phase 10T execution owns the run lock") from error
        handle.seek(0)
        handle.truncate()
        handle.write(_json_bytes({"pid": os.getpid(), "started_at": _now()}).decode())
        handle.flush()
        os.fsync(handle.fileno())
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _attempt_directory(root: Path, stage: str) -> Path:
    if stage.casefold() in FORBIDDEN_PARTS:
        raise Phase10TRunError("protected holdout stages are not executable")
    parent = _safe_path(root, RUN_ROOT / stage)
    parent.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        item.name for item in parent.iterdir() if item.is_dir() and item.name.startswith("attempt-")
    )
    number = 1
    if existing:
        number = max(int(name.removeprefix("attempt-")) for name in existing) + 1
    path = parent / f"attempt-{number:04d}"
    path.mkdir()
    return path


def _write_immutable(path: Path, value: object) -> str:
    encoded = _json_bytes(value)
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return _sha256_bytes(encoded)


def _write_text_immutable(path: Path, text: str) -> str:
    data = text.encode("utf-8")
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return _sha256_bytes(data)


def _teacher_eligible_row(row: dict[str, Any]) -> bool:
    """Return whether a prepared row can receive a normal teacher root label."""

    raw_targets = row.get("raw_targets")
    if not isinstance(raw_targets, dict) or not isinstance(
        raw_targets.get("terminal_position"), bool
    ):
        raise Phase10TRunError("approved train row has no terminal-position identity")
    legal_moves = row.get("legal_moves")
    if raw_targets["terminal_position"]:
        if legal_moves not in (None, []):
            raise Phase10TRunError("terminal train row unexpectedly has legal roots")
        return False
    if not isinstance(legal_moves, list) or not legal_moves:
        raise Phase10TRunError("non-terminal train row has no legal-root inventory")
    return True


def _load_train_rows(
    root: Path, limit: int, *, selection_stats: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    if limit <= 0:
        raise Phase10TRunError("label limit must be positive")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_index = 0
    handles = []
    try:
        for source_path in TRAIN_SOURCES:
            path = _safe_path(root, source_path)
            handle = path.open(encoding="utf-8")
            handles.append(handle)
        # Round-robin preserves a deterministic source-balanced prefix without ever opening any
        # validation, source-held-out, final, or reserved payload.
        active = list(range(len(handles)))
        while active and len(rows) < limit:
            next_active: list[int] = []
            for index in active:
                line = handles[index].readline()
                if not line:
                    continue
                next_active.append(index)
                row = json.loads(line)
                if row.get("split") != "train" or row.get("source") not in APPROVED_SOURCES:
                    raise Phase10TRunError(
                        "approved train source contains a non-train or unknown row"
                    )
                key = str(row.get("sfen", ""))
                if not key or key in seen:
                    continue
                if not _teacher_eligible_row(row):
                    if selection_stats is not None:
                        selection_stats["terminal_rows_skipped"] = (
                            selection_stats.get("terminal_rows_skipped", 0) + 1
                        )
                    continue
                seen.add(key)
                rows.append(row)
                if len(rows) >= limit:
                    break
            active = next_active
            source_index += 1
            if source_index > limit * 4 + 4:
                raise Phase10TRunError("deterministic train selection exceeded its bounded scan")
    finally:
        for handle in handles:
            handle.close()
    if len(rows) != limit:
        raise Phase10TRunError(
            f"approved train sources yielded {len(rows)} unique rows, expected {limit}"
        )
    return rows


def _candidate_dict(candidate: Any) -> dict[str, Any]:
    return {
        "multipv": candidate.multipv,
        "score": candidate.score.as_dict(),
        "pv": list(candidate.pv),
        "depth": candidate.depth,
        "seldepth": candidate.seldepth,
        "nodes": candidate.nodes,
    }


def _validate_label_result(row: dict[str, Any], result: Any) -> None:
    """Validate observed scores/roots; full PV legality is independently replayed."""

    candidates = tuple(result.candidates)
    if not 1 <= len(candidates) <= 3:
        raise Phase10TRunError("teacher MultiPV returned an invalid candidate count")
    legal_moves = row.get("legal_moves")
    if not isinstance(legal_moves, list) or not legal_moves:
        raise Phase10TRunError("approved train row has no legal-root inventory")
    first_moves = [candidate.pv[0] for candidate in candidates if candidate.pv]
    if len(first_moves) != len(candidates) or len(set(first_moves)) != len(first_moves):
        raise Phase10TRunError("teacher MultiPV contains an empty or duplicate root")
    if result.bestmove != first_moves[0] or any(move not in legal_moves for move in first_moves):
        raise Phase10TRunError("teacher MultiPV contains an illegal root")
    if any(
        isinstance(candidate.multipv, bool) or not isinstance(candidate.multipv, int)
        for candidate in candidates
    ) or [candidate.multipv for candidate in candidates] != list(range(1, len(candidates) + 1)):
        raise Phase10TRunError("teacher MultiPV ranks are inconsistent")
    try:
        for candidate in candidates:
            score_order(candidate.score.as_dict())
    except ValueError as error:
        raise Phase10TRunError(f"teacher MultiPV inconsistent-score: {error}") from error


def _analyze_teacher_label(
    teacher: Any,
    row: dict[str, Any],
    nodes: int,
    *,
    retry_stats: dict[str, Any] | None = None,
) -> tuple[Any, int]:
    """Accept up to three valid observed candidates without coverage-only retries."""
    result = teacher.analyze_with_retry(str(row["sfen"]), nodes=nodes)
    _validate_label_result(row, result)
    return result, 0


def _label_manifest_row(index: int, row: dict[str, Any], label: dict[str, Any]) -> dict[str, Any]:
    if (
        label.get("schema") != LABEL_SCHEMA
        or label.get("index") != index
        or label.get("sfen") != row.get("sfen")
        or label.get("source") != row.get("source")
        or label.get("split") != "train"
        or label.get("represented_position") != row.get("sfen")
    ):
        raise Phase10TRunError("resumed label prefix does not match the deterministic train rows")
    teacher = label.get("teacher")
    if not isinstance(teacher, dict) or teacher.get("nodes") not in {100_000, 400_000}:
        raise Phase10TRunError("resumed label prefix has an invalid teacher identity")
    candidates = teacher.get("candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 3:
        raise Phase10TRunError("resumed label prefix has an invalid MultiPV result")
    try:
        result = USISearchResult(
            bestmove=teacher["bestmove"],
            candidates=tuple(
                USICandidate(
                    multipv=candidate["multipv"],
                    score=USIScore(**candidate["score"]),
                    pv=tuple(candidate["pv"]),
                    depth=candidate["depth"],
                    seldepth=candidate["seldepth"],
                    nodes=candidate["nodes"],
                )
                for candidate in candidates
            ),
            elapsed_ms=teacher["elapsed_ms"],
        )
        _validate_label_result(row, result)
    except (KeyError, TypeError, ValueError) as error:
        raise Phase10TRunError("resumed label prefix has corrupt candidates") from error
    score_kind = teacher.get("primary_score_kind")
    if (
        score_kind != result.primary.score.kind
        or teacher.get("primary_score_value") != result.primary.score.value
        or isinstance(teacher.get("primary_score_value"), bool)
        or teacher.get("multipv") != 3
        or not isinstance(teacher.get("name"), str)
        or not teacher["name"]
    ):
        raise Phase10TRunError("resumed label prefix has an inconsistent primary score or identity")
    if label.get("wdl_mask") is not bool(row.get("wdl_mask")) or label.get("wdl") != (
        row.get("wdl") if row.get("wdl_mask") else None
    ):
        raise Phase10TRunError("resumed label prefix factual WDL identity changed")
    try:
        targets = observed_targets(candidates, factual_wdl=label.get("wdl"))
    except ValueError as error:
        raise Phase10TRunError(str(error)) from error
    if "observed_targets" in label and label["observed_targets"] != targets:
        raise Phase10TRunError("resumed label prefix target masks changed")
    validation_retries = teacher.get("validation_retries", 0)
    if (
        isinstance(validation_retries, bool)
        or not isinstance(validation_retries, int)
        or not 0 <= validation_retries <= MULTIPV_VALIDATION_RETRIES
    ):
        raise Phase10TRunError("resumed label prefix has an invalid validation retry count")
    return {
        "index": index,
        "sfen_sha256": _sha256_bytes(str(row["sfen"]).encode()),
        "source": row["source"],
        "score_kind": score_kind,
        "nodes": teacher["nodes"],
        "validation_retries": validation_retries,
        "observed_targets": targets,
    }


def _resume_label_prefix(
    root: Path,
    rung: str,
    nodes: int,
    rows: Sequence[dict[str, Any]],
    *,
    validator: Any | None = None,
) -> tuple[bytes, list[dict[str, Any]], str | None]:
    """Reuse only a verified prefix from the latest failed immutable label attempt."""

    parent = _safe_path(root, RUN_ROOT / f"labels-{rung}")
    if not parent.is_dir():
        return b"", [], None
    for attempt in sorted(
        (path for path in parent.iterdir() if path.is_dir() and path.name.startswith("attempt-")),
        reverse=True,
    ):
        receipt_path = attempt / "receipt.json"
        if not receipt_path.is_file():
            continue
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if not isinstance(receipt, dict):
                raise Phase10TRunError("resumed label prefix receipt is corrupt")
            if receipt.get("status") != "failed" or receipt.get("nodes") != nodes:
                continue
            if receipt.get("teacher") != _teacher_identity(root):
                raise Phase10TRunError("resumed label prefix teacher identity changed")
            relative = Path(str(receipt["labels_path"]))
            labels_path = _safe_path(root, relative)
            if receipt.get("labels_sha256") != _sha256_file(labels_path):
                raise Phase10TRunError("resumed label prefix content hash changed")
            raw = labels_path.read_bytes()
            if raw and not raw.endswith(b"\n"):
                raise Phase10TRunError("resumed label prefix has an incomplete final line")
            labels = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
            if any(
                not isinstance(label, dict) or not isinstance(label.get("teacher"), dict)
                for label in labels
            ):
                raise Phase10TRunError("resumed label prefix contains a corrupt label object")
            completed = receipt.get("positions_completed")
            if (
                type(completed) is not int
                or completed != len(labels)
                or not 0 <= completed < len(rows)
            ):
                raise Phase10TRunError("resumed label prefix completed count is corrupt")
            if len({label.get("teacher", {}).get("name") for label in labels}) > 1:
                raise Phase10TRunError("resumed label prefix runtime teacher identity changed")
            manifest = [
                _label_manifest_row(index, rows[index], label) for index, label in enumerate(labels)
            ]
            with contextlib.ExitStack() as stack:
                replay = validator or stack.enter_context(RustLegalityValidator(root))
                for label in labels:
                    teacher = label["teacher"]
                    if teacher["nodes"] != nodes:
                        raise Phase10TRunError("resumed label prefix node budget changed")
                    replay.validate(
                        label["sfen"],
                        teacher["bestmove"],
                        [candidate["pv"] for candidate in teacher["candidates"]],
                        configured_multipv=3,
                    )
            return raw, manifest, relative.as_posix()
        except (OSError, TypeError, ValueError, KeyError) as error:
            raise Phase10TRunError(
                f"resumed label prefix is corrupt or unreadable at {attempt.name}: {error}"
            ) from error
    return b"", [], None


def _label(root: Path, base: dict[str, Any], rung: str, limit: int, nodes: int) -> dict[str, Any]:
    if rung not in RUNG_LIMITS:
        raise Phase10TRunError(f"unsupported label rung: {rung}")
    required = RUNG_LIMITS[rung]
    if limit != required and rung != "micro":
        raise Phase10TRunError(f"rung {rung} requires exactly {required} selected positions")
    attempt = _attempt_directory(root, f"labels-{rung}")
    labels_path = attempt / "labels.jsonl"
    manifest_rows: list[dict[str, Any]] = []
    config = load_teacher_config(_safe_path(root, Path("configs/teacher/apery-v2.0.0.yaml")))
    started = time.monotonic()
    resumed_from: str | None = None
    validator_identity: dict[str, object] | None = None
    selection_stats = {
        "terminal_rows_skipped": 0,
        "teacher_validation_retries": 0,
        "teacher_validation_retry_limit": 0,
        "historical_validation_retry_limit": MULTIPV_VALIDATION_RETRIES,
        "rule": "nonterminal_with_nonempty_legal_root_inventory_v1",
    }
    try:
        rows = _load_train_rows(root, limit, selection_stats=selection_stats)
        prefix, manifest_rows, resumed_from = _resume_label_prefix(root, rung, nodes, rows)
        with (
            labels_path.open("x", encoding="utf-8", newline="") as output,
            USIEngine(config, root) as teacher,
            RustLegalityValidator(root) as validator,
        ):
            if prefix:
                output.write(prefix.decode("utf-8"))
                output.flush()
            validator_identity = validator.identity.as_dict() if validator.identity else None
            identity = teacher.identity
            if identity is None:
                raise Phase10TRunError("teacher did not publish a runtime identity")
            if prefix and json.loads(prefix.splitlines()[0])["teacher"]["name"] != identity.name:
                raise Phase10TRunError("resumed label prefix runtime teacher identity changed")
            for index in range(len(manifest_rows), len(rows)):
                row = rows[index]
                try:
                    result, validation_retries = _analyze_teacher_label(
                        teacher, row, nodes, retry_stats=selection_stats
                    )
                except Phase10TRunError as error:
                    raise Phase10TRunError(
                        f"invalid teacher result at row {index}: {error}"
                    ) from error
                validator.validate(
                    row["sfen"],
                    result.bestmove,
                    [list(candidate.pv) for candidate in result.candidates],
                    configured_multipv=3,
                )
                primary = result.primary.score
                payload = {
                    "schema": LABEL_SCHEMA,
                    "index": index,
                    "sfen": row["sfen"],
                    "source": row["source"],
                    "split": row["split"],
                    "wdl": row.get("wdl") if row.get("wdl_mask") else None,
                    "wdl_mask": bool(row.get("wdl_mask")),
                    "represented_position": row["sfen"],
                    "teacher": {
                        "name": identity.name,
                        "nodes": nodes,
                        "multipv": 3,
                        "bestmove": result.bestmove,
                        "candidates": [
                            _candidate_dict(candidate) for candidate in result.candidates
                        ],
                        "primary_score_kind": primary.kind,
                        "primary_score_value": primary.value,
                        "elapsed_ms": result.elapsed_ms,
                        "validation_retries": validation_retries,
                    },
                }
                payload["observed_targets"] = observed_targets(
                    payload["teacher"]["candidates"], factual_wdl=payload["wdl"]
                )
                manifest_row = _label_manifest_row(index, row, payload)
                output.write(_json_bytes(payload).decode())
                output.flush()
                manifest_rows.append(manifest_row)
    except Exception as error:
        partial_hash = _sha256_file(labels_path) if labels_path.is_file() else None
        failure_receipt = {
            "schema": RECEIPT_SCHEMA,
            "status": "failed",
            "stage": "teacher_labeling",
            "rung": rung,
            "nodes": nodes,
            "positions_completed": len(manifest_rows),
            "labels_path": labels_path.relative_to(root).as_posix(),
            "labels_sha256": partial_hash,
            "teacher": base["teacher"],
            "legality_validator": validator_identity,
            "selection": selection_stats,
            "error": str(error),
        }
        failure_path = attempt / "receipt.json"
        failure_receipt["receipt_path"] = failure_path.relative_to(root).as_posix()
        failure_receipt["receipt_sha256"] = _write_immutable(failure_path, failure_receipt)
        raise Phase10TRunError(
            f"teacher labeling failed after {len(manifest_rows)} rows: {error}"
        ) from error
    labels_sha256 = _sha256_file(labels_path)
    manifest = {
        "schema": "open_shogiai_phase10t_label_manifest/v1",
        "rung": rung,
        "positions": len(manifest_rows),
        "nodes": nodes,
        "teacher": base["teacher"],
        "legality_validator": validator_identity,
        "selection": selection_stats,
        "labels_path": labels_path.relative_to(root).as_posix(),
        "labels_sha256": labels_sha256,
        "rows": manifest_rows,
        "elapsed_seconds": time.monotonic() - started,
        "resumed_from": resumed_from,
    }
    manifest_path = attempt / "manifest.json"
    manifest_sha256 = _write_immutable(manifest_path, manifest)
    return {
        "status": "complete",
        "stage": "teacher_labeling",
        "rung": rung,
        "positions": len(manifest_rows),
        "nodes": nodes,
        "labels_path": labels_path.relative_to(root).as_posix(),
        "labels_sha256": labels_sha256,
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "manifest_sha256": manifest_sha256,
        "resumed_from": resumed_from,
    }


def _load_labels(root: Path, labels_path: Path) -> list[dict[str, Any]]:
    path = _safe_path(root, labels_path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != LABEL_SCHEMA or row.get("split") != "train":
                raise Phase10TRunError(f"invalid or non-train label row at {path}:{line_number}")
            if row.get("represented_position") != row.get("sfen"):
                raise Phase10TRunError("root/represented position identity changed")
            rows.append(row)
    if not rows:
        raise Phase10TRunError("label file contains no rows")
    return rows


def _campaign_examples(
    root: Path, labels: Sequence[dict[str, Any]], *, expected_nodes: int
) -> list[Phase10TExample]:
    approved = _load_train_rows(root, len(labels))
    if len(approved) != len(labels):
        raise Phase10TRunError("label population does not match the approved train population")
    examples: list[Phase10TExample] = []
    seen: set[str] = set()
    for index, (label, approved_row) in enumerate(zip(labels, approved, strict=True)):
        if (
            label.get("sfen") != approved_row.get("sfen")
            or label.get("source") != approved_row.get("source")
            or label.get("split") != "train"
        ):
            raise Phase10TRunError(f"label lineage differs from approved train row {index}")
        sfen = str(label["sfen"])
        if sfen in seen:
            raise Phase10TRunError("label lineage contains duplicate positions")
        seen.add(sfen)
        _label_manifest_row(index, approved_row, label)
        teacher = label.get("teacher")
        if not isinstance(teacher, dict) or teacher.get("nodes") != expected_nodes:
            raise Phase10TRunError("label lineage has an unexpected teacher node budget")
        kind = teacher.get("primary_score_kind")
        value = teacher.get("primary_score_value")
        if kind not in {"cp", "mate"} or not isinstance(value, int):
            raise Phase10TRunError(f"label lineage has an invalid teacher score at row {index}")
        examples.append(
            Phase10TExample(
                sfen=sfen,
                cp=float(value) if kind == "cp" else None,
                wdl=int(label["wdl"])
                if label.get("wdl_mask") and label.get("wdl") in (0, 1, 2)
                else None,
                history=HistoryFacts(),
                source=str(label["source"]),
            )
        )
    return examples


def _fit_cp_calibration(
    model: Any, validation_rows: Sequence[Phase10TExample], seed: int
) -> dict[str, float | int]:
    cp_rows = [row for row in validation_rows if row.cp is not None]
    if not cp_rows:
        raise Phase10TRunError("validation partition has no direct teacher cp labels")
    predictions = predict_examples(model, cp_rows)[:, 0].astype(np.float64)
    targets = np.asarray([float(row.cp) for row in cp_rows], dtype=np.float64)
    if len(cp_rows) >= 2 and float(np.var(predictions)) > 1e-12:
        design = np.column_stack((predictions, np.ones(len(predictions))))
        scale, bias = np.linalg.lstsq(design, targets, rcond=None)[0]
    else:
        scale, bias = 1.0, float(np.mean(targets - predictions))
    if not np.isfinite(scale) or not np.isfinite(bias) or scale <= 0.0:
        raise Phase10TRunError("teacher-cp calibration produced a non-positive affine scale")
    model.head_weight[:, 0] *= np.float32(scale)
    model.head_bias[0] = np.float32(float(model.head_bias[0]) * scale + bias)
    model.validate()
    calibrated = predict_examples(model, cp_rows)[:, 0].astype(np.float64)
    return {
        "scale": float(scale),
        "bias": float(bias),
        "examples": len(cp_rows),
        "mae_before": float(np.mean(np.abs(predictions - targets))),
        "mae_after": float(np.mean(np.abs(calibrated - targets))),
        "seed": seed,
        "target": "approved_teacher_cp_only",
    }


def _train(
    root: Path,
    base: dict[str, Any],
    rung: str,
    labels_path: Path,
    variant: str,
) -> dict[str, Any]:
    if rung == "micro":
        rows = _load_labels(root, labels_path)
        if len(rows) != RUNG_LIMITS["micro"]:
            raise Phase10TRunError("micro-overfit requires exactly 32 labeled positions")
        examples: list[Phase10TExample] = []
        for row in rows:
            teacher = row["teacher"]
            if teacher.get("primary_score_kind") != "cp":
                raise Phase10TRunError(
                    "micro-overfit requires direct cp teacher labels; mate rows are symbolic only"
                )
            examples.append(
                Phase10TExample(
                    sfen=str(row["sfen"]),
                    cp=float(teacher["primary_score_value"]),
                    wdl=int(row["wdl"])
                    if row.get("wdl_mask") and row.get("wdl") in (0, 1, 2)
                    else None,
                    history=HistoryFacts(),
                )
            )
        model, stats = train_random_lineage(
            examples,
            seed=DEFAULT_SEED,
            max_steps=2_000,
            time_limit_seconds=120.0,
        )
        attempt = _attempt_directory(root, "micro")
        model_path = attempt / "a1-micro.osat10"
        model_sha256 = model.write(model_path)
        receipt = {
            "status": "complete",
            "stage": "micro_overfit",
            "rung": "micro",
            "input_labels": labels_path.as_posix(),
            "input_labels_sha256": _sha256_file(_safe_path(root, labels_path)),
            "model_path": model_path.relative_to(root).as_posix(),
            "model_sha256": model_sha256,
            "model_artifact_sha256": model_sha256,
            "random_seed": DEFAULT_SEED,
            "stats": stats,
            "campaign_counted": False,
            "weights_discard_before_100k": True,
            "integrity": {"native_a1": True, "target_semantics": True, "holdout_free": True},
        }
        receipt_path = attempt / "receipt.json"
        receipt["receipt_path"] = receipt_path.relative_to(root).as_posix()
        receipt["receipt_sha256"] = _write_immutable(receipt_path, receipt)
        return receipt
    if rung not in {"100k", "1m"} or variant != "a1-king-relative-128":
        raise Phase10TRunError("only the frozen a1 campaign variant is executable here")
    audit = _pure_build_audit(root)
    if not audit["passed"]:
        raise Phase10TRunError(
            "supervised campaign training is STOP_CLOSED without the pure-only audit"
        )
    rows = _load_labels(root, labels_path)
    required = RUNG_LIMITS[rung]
    if len(rows) != required:
        raise Phase10TRunError(f"{rung} training requires exactly {required} labeled positions")
    seed = {"100k": 2_026_0907, "1m": 2_026_0908}[rung]
    examples = _campaign_examples(root, rows, expected_nodes=100_000)
    pre_model, selected_model, last_model, stats = train_supervised_lineage(
        examples,
        seed=seed,
        max_passes=2,
        batch_size=128,
        validation_every_steps=250,
        patience=4,
        learning_rate=3e-4,
        min_learning_rate=3e-5,
        weight_decay=1e-4,
        gradient_clip=5.0,
    )
    order = np.random.default_rng(seed).permutation(len(examples))
    validation_rows = [examples[index] for index in order[: max(1, len(examples) // 10)]]
    calibration = _fit_cp_calibration(selected_model, validation_rows, seed)
    attempt = _attempt_directory(root, f"train-{rung}")
    pre_path = attempt / "pre-stage.osat10"
    last_path = attempt / "last.osat10"
    selected_path = attempt / f"selected-{variant}.osat10"
    pre_sha256 = pre_model.write(pre_path)
    last_sha256 = last_model.write(last_path)
    selected_sha256 = selected_model.write(selected_path)
    receipt = {
        "status": "complete",
        "stage": "supervised_training",
        "rung": rung,
        "variant": variant,
        "input_labels": _safe_path(root, labels_path).relative_to(root).as_posix(),
        "input_labels_sha256": _sha256_file(_safe_path(root, labels_path)),
        "random_seed": seed,
        "checkpoints": {
            "pre_stage": {"path": pre_path.relative_to(root).as_posix(), "sha256": pre_sha256},
            "last": {"path": last_path.relative_to(root).as_posix(), "sha256": last_sha256},
            "selected": {
                "path": selected_path.relative_to(root).as_posix(),
                "sha256": selected_sha256,
            },
        },
        "model_path": selected_path.relative_to(root).as_posix(),
        "model_sha256": selected_sha256,
        "model_artifact_sha256": selected_sha256,
        "stats": stats,
        "calibration": calibration,
        "campaign_counted": True,
        "holdout_access": False,
        "pure_only_build_audit": audit,
        "integrity": {
            "approved_train_only": True,
            "unique_positions": True,
            "source_provenance": True,
            "target_semantics": True,
            "native_a1": False,
            "native_wasm_parity": False,
            "pure_runtime": False,
        },
    }
    receipt_path = attempt / "receipt.json"
    receipt["receipt_path"] = receipt_path.relative_to(root).as_posix()
    receipt["receipt_sha256"] = _write_immutable(receipt_path, receipt)
    return receipt


def _diagnostics(root: Path, base: dict[str, Any]) -> dict[str, Any]:
    commands = [
        [sys.executable, "-m", "open_shogi_training.phase10t_accumulator"],
        ["cargo", "test", "--locked", "-p", "open-shogi-core", "phase10t", "--lib"],
        ["cargo", "test", "--locked", "-p", "open-shogi-core", "pure_learned", "--lib"],
    ]
    outputs = []
    for command in commands:
        completed = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=300)
        outputs.append(
            {
                "command": command,
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
            }
        )
        if completed.returncode:
            raise Phase10TRunError(f"diagnostic command failed: {' '.join(command)}")
    pure_build = json.loads(_safe_path(root, Path("configs/phase10t/pure-build.json")).read_text())
    return {
        "status": "complete",
        "stage": "diagnostics",
        "commands": outputs,
        "frozen_pure_build_status_detail": pure_build["status_detail"],
        "native_a1_loader_tested": True,
        "actual_wasm_tested": _pure_build_audit(root)["actual_wasm_tested"],
        "pure_only_build_audit": _pure_build_audit(root),
    }


def _pure_build_audit(root: Path) -> dict[str, Any]:
    """Rehash the implementation and actual artifacts, never trust design status alone."""
    from open_shogi_training.phase10t_pure_build import verify_evidence

    control = json.loads(_safe_path(root, Path("configs/phase10t/pure-build.json")).read_text())
    result = {
        "passed": False,
        "control_status_detail": control.get("status_detail"),
        "actual_wasm_tested": False,
    }
    if control.get("status") != "implementation_verified":
        return result
    try:
        binding = control["evidence"]
        path = _safe_path(root, Path(binding["path"]))
        if phase10t.sha256(path) != binding["sha256"]:
            raise ValueError("pure-only artifact audit receipt hash mismatch")
        receipt = json.loads(path.read_text())
        verify_evidence(root, receipt)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["reason"] = str(error)
        return result
    return {
        **result,
        "passed": True,
        "actual_wasm_tested": True,
        "audit_git_commit": receipt["git_commit"],
        "receipt_sha256": binding["sha256"],
    }


def _crossplay(root: Path, base: dict[str, Any], model_path: Path) -> dict[str, Any]:
    audit = _pure_build_audit(root)
    if not audit["passed"]:
        raise Phase10TRunError(
            "train-only cross-play is STOP_CLOSED: frozen pure-only build audit is incomplete"
        )
    if not model_path.is_file():
        raise Phase10TRunError("cross-play model is missing")
    raise Phase10TRunError(
        "train-only c0/a1 cross-play wiring is not proven for this model lineage"
    )


def _arena(root: Path, base: dict[str, Any], model_path: Path, gate: str) -> dict[str, Any]:
    audit = _pure_build_audit(root)
    if not audit["passed"]:
        raise Phase10TRunError(
            "Arena is STOP_CLOSED: pure-only compile/link/Wasm audit is incomplete; "
            "no strength evidence may be recorded"
        )
    if gate not in {"recovery", "viable", "selfplay_entry", "final_objective"}:
        raise Phase10TRunError(f"unknown frozen Arena gate: {gate}")
    if gate in {"selfplay_entry", "final_objective"}:
        raise Phase10TRunError(
            "self-play and final objective execution are outside this review boundary"
        )
    if not model_path.is_file():
        raise Phase10TRunError("Arena model is missing")
    raise Phase10TRunError(
        "Arena schedule requires the completed 100k/1M lineage and c0 control receipts"
    )


def _report(root: Path, base: dict[str, Any]) -> dict[str, Any]:
    run_root = _safe_path(root, RUN_ROOT)
    receipts = []
    for path in sorted(run_root.glob("*/attempt-*/receipt.json")):
        if path.is_symlink():
            raise Phase10TRunError(f"symlinked receipt: {path}")
        receipts.append(path.relative_to(root).as_posix())
    return {
        "status": "review_required",
        "stage": "review_gate",
        "receipts": receipts,
        "final_holdout_inspected": False,
        "promotion_performed": False,
        "selfplay_started": False,
        "merge_push_release_deploy_performed": False,
        "open_shogi_ui_modified": False,
    }


def _dry_run(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    return {
        "status": "dry_run",
        "stage": args.stage,
        "rung": args.rung,
        "bounded": True,
        "writes": [],
        "teacher_calls": 0,
        "arena_games": 0,
        "selfplay_games": 0,
        "holdout_access": "forbidden",
        "planned": {
            "preflight": (
                "validate frozen manifest, current commit, clean worktree, resources, "
                "implementation and teacher identities"
            ),
            "micro": (
                "label exactly 32 approved train positions then train <=2000 steps/120 seconds"
            ),
            "arena": (
                "requires pure-only build audit and frozen 48% entry gate before any self-play"
            ),
        },
    }


def _execute(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    base = _base_identity(root)
    if args.stage == "preflight":
        return {
            **base,
            "status": "ready",
            "stage": "preflight",
            "pure_only_build_audit": _pure_build_audit(root),
        }
    if args.stage == "diagnostics":
        return {**base, **_diagnostics(root, base)}
    if args.stage == "label":
        nodes = args.nodes if not args.hard else args.hard_nodes
        return {
            **base,
            **_label(root, base, args.rung, args.limit or RUNG_LIMITS[args.rung], nodes),
        }
    if args.stage == "train":
        if args.labels is None:
            raise Phase10TRunError("--labels is required for training")
        return {**base, **_train(root, base, args.rung, args.labels, args.variant)}
    if args.stage == "crossplay":
        if args.model is None:
            raise Phase10TRunError("--model is required for cross-play")
        return {**base, **_crossplay(root, base, args.model)}
    if args.stage == "arena":
        if args.model is None:
            raise Phase10TRunError("--model is required for Arena")
        return {**base, **_arena(root, base, args.model, args.gate)}
    if args.stage == "report":
        return {**base, **_report(root, base)}
    raise Phase10TRunError(f"unsupported stage: {args.stage}")


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit("Closed campaign; use current development commands")


if __name__ == "__main__":
    main()
