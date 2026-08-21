"""Format adapters and source-preserving records for Phase 10R.

The adapters intentionally separate byte decoding from board legality.  CSA
records can be replayed by the existing Rust rule engine; KIF and fixed binary
formats remain explicitly marked as requiring their decoder/replay stage until
that stage has produced a canonical board identity.  Unknown score semantics
are retained verbatim and never converted into training targets here.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import lzma
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from open_shogi_training.data.external_audit import (
    parse_csa_sample,
    parse_hcpe3_sample,
    parse_hcpe_sample,
    parse_kif_sample,
    parse_packed_sfen_value_sample,
)
from open_shogi_training.data.phase10r_kif import convert_kif_to_csa

PHASE10R_NORMALIZED_SCHEMA = "phase10r_normalized_record/v1"
MAX_SOURCE_BYTES = 64 * 1024 * 1024
MAX_DECOMPRESSED_BYTES = 256 * 1024 * 1024


class Phase10RAdapterError(ValueError):
    """Raised when a source format cannot be adapted without guessing."""


def normalize_source_bytes(
    raw: bytes,
    *,
    format_name: str,
    source_id: str,
    artifact_id: str,
    source_revision: str = "unknown",
    license_decision: Mapping[str, Any] | None = None,
    max_bytes: int = MAX_SOURCE_BYTES,
    max_records: int = 100_000,
) -> list[dict[str, Any]]:
    """Adapt one bounded object while retaining all source-specific fields."""

    if len(raw) > max_bytes:
        raise Phase10RAdapterError(f"source sample exceeds {max_bytes} bytes")
    normalized_format = format_name.lower().lstrip(".")
    parser = _parser_for(normalized_format)
    records = parser(
        raw,
        source_id=source_id,
        artifact_id=artifact_id,
        license_decision=license_decision,
        max_bytes=max_bytes,
        max_records=max_records,
    )
    return [
        _upgrade_record(
            record,
            source_revision=source_revision,
            legality_method=_legality_method(normalized_format),
        )
        for record in records
    ]


def normalize_source_file(
    path: Path,
    *,
    source_id: str,
    artifact_id: str,
    format_name: str | None = None,
    compression: str | None = None,
    source_revision: str = "unknown",
    license_decision: Mapping[str, Any] | None = None,
    max_bytes: int = MAX_SOURCE_BYTES,
    max_records: int = 100_000,
) -> list[dict[str, Any]]:
    """Read one sample file, with bounded XZ decompression, then adapt it."""

    raw = _read_bounded(path, max_bytes=max_bytes)
    selected_format = format_name or _format_from_path(path)
    if compression == "xz" or path.suffix.lower() == ".xz":
        raw = _decompress_xz(raw, max_bytes=MAX_DECOMPRESSED_BYTES)
    return normalize_source_bytes(
        raw,
        format_name=selected_format,
        source_id=source_id,
        artifact_id=artifact_id,
        source_revision=source_revision,
        license_decision=license_decision,
        max_bytes=max_bytes,
        max_records=max_records,
    )


def write_normalized_jsonl(records: Iterable[Mapping[str, Any]], output: Path) -> dict[str, Any]:
    """Write deterministic, gzip-compressed normalized records once."""

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise Phase10RAdapterError(f"refusing to overwrite normalized output: {output}")
    count = 0
    digest = hashlib.sha256()
    with (
        output.open("xb") as raw_handle,
        gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0, filename="") as handle,
    ):
        for record in records:
            line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            encoded = (line + "\n").encode("utf-8")
            handle.write(encoded)
            digest.update(encoded)
            count += 1
    return {
        "schema": "phase10r_normalized_jsonl_manifest/v1",
        "record_count": count,
        "output": output.name,
        "sha256_uncompressed_jsonl": digest.hexdigest(),
        "size_bytes": output.stat().st_size,
    }


def validate_csa_with_engine(
    cli: Path,
    input_dir: Path,
    output: Path,
    *,
    max_games: int = 100,
) -> dict[str, Any]:
    """Replay bounded CSA samples with the repository's Rust rule engine."""

    if not cli.is_file() or not input_dir.is_dir():
        raise Phase10RAdapterError("CSA validation requires a regular CLI and input directory")
    if output.exists():
        raise Phase10RAdapterError(f"refusing to overwrite engine validation output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staged_input = output.parent / f".{output.stem}-utf8-csa"
    _stage_csa_utf8(input_dir, staged_input)
    command = [
        str(cli),
        "export-csa-jsonl",
        "--input-dir",
        str(staged_input),
        "--output",
        str(output),
        "--max-games",
        str(max_games),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise Phase10RAdapterError(
            "Rust CSA legality validation failed: "
            + (completed.stderr.strip() or completed.stdout.strip() or "unknown error")
        )
    rows: list[dict[str, Any]] = []
    for line in output.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise Phase10RAdapterError("Rust CSA validator emitted invalid JSONL") from error
        if not isinstance(value, dict):
            raise Phase10RAdapterError("Rust CSA validator emitted a non-object row")
        rows.append(value)
    accepted = sum(1 for row in rows if row.get("status") in {"accepted", "ok", "valid"})
    rejected = len(rows) - accepted
    return {
        "schema": "phase10r_legality_report/v1",
        "format": "csa",
        "method": "open-shogi-cli export-csa-jsonl",
        "input_dir": input_dir.name,
        "rows": len(rows),
        "accepted": accepted,
        "rejected": rejected,
        "all_accepted": bool(rows) and rejected == 0,
        "details": rows,
    }


def validate_kif_with_engine(
    cli: Path,
    kif_path: Path,
    output: Path,
    *,
    max_moves: int = 10_000,
) -> dict[str, Any]:
    """Convert one bounded KIF game and replay its derived CSA with Rust."""

    if output.exists():
        raise Phase10RAdapterError(f"refusing to overwrite engine validation output: {output}")
    raw = _read_bounded(kif_path, max_bytes=MAX_SOURCE_BYTES)
    csa, conversion = convert_kif_to_csa(raw, max_moves=max_moves)
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.parent / f".{output.stem}-kif-source"
    if staged.exists():
        raise Phase10RAdapterError(f"refusing to overwrite KIF staging directory: {staged}")
    staged.mkdir(parents=True)
    (staged / "converted.csa").write_text(csa, encoding="utf-8")
    report = validate_csa_with_engine(cli, staged, output, max_games=1)
    report["format"] = "kif"
    report["method"] = "kif-to-csa-then-open-shogi-cli"
    report["conversion"] = conversion
    return report


def _stage_csa_utf8(input_dir: Path, staged_input: Path) -> None:
    """Create an immutable derived UTF-8 view for the Rust CSA reader."""

    if staged_input.exists():
        raise Phase10RAdapterError(f"refusing to overwrite CSA staging directory: {staged_input}")
    staged_input.mkdir(parents=True)
    files = sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".csa"
    )
    if not files:
        raise Phase10RAdapterError("CSA input directory contains no regular .csa files")
    for source in files:
        try:
            text = source.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = source.read_bytes().decode("cp932")
            except UnicodeDecodeError as error:
                raise Phase10RAdapterError(
                    f"CSA source is not UTF-8 or CP932: {source.name}"
                ) from error
        lines = text.splitlines()
        if lines and lines[0].startswith("'CSA encoding="):
            lines[0] = "'CSA encoding=UTF-8"
        else:
            lines.insert(0, "'CSA encoding=UTF-8")
        if len(lines) > 1 and lines[1] == "V2.2":
            lines[1] = "V3.0"
        text = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
        destination = staged_input / source.name
        destination.write_text(text, encoding="utf-8")


def _parser_for(format_name: str):
    if format_name == "csa":
        return parse_csa_sample
    if format_name == "kif":
        return parse_kif_sample
    if format_name == "hcpe":
        return parse_hcpe_sample
    if format_name == "hcpe3":
        return parse_hcpe3_sample
    if format_name in {"packed_sfen_value", "binpack", "bin"}:
        return lambda raw, **kwargs: parse_packed_sfen_value_sample(
            raw, format_name=format_name, **kwargs
        )
    raise Phase10RAdapterError(f"unsupported Phase 10R format: {format_name}")


def _upgrade_record(
    record: Mapping[str, Any], *, source_revision: str, legality_method: str
) -> dict[str, Any]:
    labels = dict(record.get("labels", {}))
    source_raw_evaluation = labels.get("raw_evaluation")
    position_identity = dict(record.get("position_identity", {}))
    history_identity = dict(record.get("history_identity", {}))
    provenance = dict(record.get("provenance", {}))
    provenance["parser_version"] = "phase10r-adapters/v1"
    provenance["source_revision"] = source_revision
    source_artifact = dict(record.get("source_artifact", {}))
    source_artifact["source_revision"] = source_revision
    raw_evaluation = {
        "value": labels.get("raw_source_score"),
        "semantics": labels.get("source_score_semantics"),
        "perspective": labels.get("score_perspective"),
        "mate": labels.get("mate_representation"),
    }
    if isinstance(source_raw_evaluation, Mapping):
        for key in ("raw", "pv"):
            if key in source_raw_evaluation:
                raw_evaluation[key] = source_raw_evaluation[key]
    return {
        "schema": PHASE10R_NORMALIZED_SCHEMA,
        "record_id": record.get("record_id"),
        "source_artifact": source_artifact,
        "position": {
            "identity": position_identity,
            "canonical_sfen": None,
            "side_to_move": None,
            "raw_position": None,
        },
        "history": {"identity": history_identity, "moves": None},
        "played_move": labels.get("played_move"),
        "best_move": labels.get("best_move"),
        "legal_move_map": None,
        "policy_visits": labels.get("policy_distribution"),
        "wdl": labels.get("wdl"),
        "raw_evaluation": raw_evaluation,
        "search": dict(record.get("search", {})),
        "source_labels": labels,
        "provenance": provenance,
        "license_decision": dict(record.get("license_decision", {})),
        "validation": {
            "status": "not_replayed",
            "method": legality_method,
            "reason": "This adapter preserves bytes; rule-engine replay is a separate gate.",
        },
    }


def _legality_method(format_name: str) -> str:
    if format_name == "csa":
        return "open-shogi-cli export-csa-jsonl"
    if format_name == "kif":
        return "kif-to-csa-then-open-shogi-cli-rule-engine"
    if format_name in {"hcpe", "hcpe3", "packed_sfen_value", "binpack", "bin"}:
        return "binary-position-decoder-and-rule-engine-required"
    return "format-specific-validator-required"


def _format_from_path(path: Path) -> str:
    suffixes = [suffix.lower().lstrip(".") for suffix in path.suffixes]
    if "hcpe3" in suffixes:
        return "hcpe3"
    if "hcpe" in suffixes:
        return "hcpe"
    if path.suffix.lower() in {".csa", ".kif", ".bin", ".binpack"}:
        return path.suffix.lower().lstrip(".")
    raise Phase10RAdapterError(f"cannot infer source format from {path.name}")


def _read_bounded(path: Path, *, max_bytes: int) -> bytes:
    if not path.is_file():
        raise Phase10RAdapterError(f"source sample is not a regular file: {path}")
    if path.stat().st_size > max_bytes:
        raise Phase10RAdapterError(f"source sample exceeds {max_bytes} bytes")
    return path.read_bytes()


def _decompress_xz(raw: bytes, *, max_bytes: int) -> bytes:
    decompressor = lzma.LZMADecompressor()
    output = bytearray()
    offset = 0
    while offset < len(raw):
        chunk = decompressor.decompress(raw[offset:], max_length=max_bytes - len(output))
        output.extend(chunk)
        consumed = len(raw[offset:]) - len(decompressor.unused_data)
        offset += consumed
        if len(output) > max_bytes:
            raise Phase10RAdapterError("XZ sample exceeds decompressed byte limit")
        if decompressor.eof:
            if decompressor.unused_data:
                offset = len(raw) - len(decompressor.unused_data)
                decompressor = lzma.LZMADecompressor()
                continue
            break
        if consumed == 0 and not chunk:
            raise Phase10RAdapterError("truncated XZ sample")
    return bytes(output)


__all__ = [
    "MAX_DECOMPRESSED_BYTES",
    "MAX_SOURCE_BYTES",
    "PHASE10R_NORMALIZED_SCHEMA",
    "Phase10RAdapterError",
    "normalize_source_bytes",
    "normalize_source_file",
    "validate_csa_with_engine",
    "validate_kif_with_engine",
    "write_normalized_jsonl",
]
