"""Deterministic overlap, deduplication, and split-leakage accounting."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any


def measure_phase10r_overlap(
    records_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Measure exact identities and report which identity layers are unavailable."""

    groups = {
        source: {
            "raw_record": _keys(
                records, lambda record: record.get("provenance", {}).get("raw_record_sha256")
            ),
            "position": _keys(
                records,
                lambda record: _nested(record, "position", "identity"),
            ),
            "history": _keys(records, lambda record: _nested(record, "history", "identity")),
            "canonical_sfen": _keys(
                records,
                lambda record: _nested(record, "position", "canonical_sfen"),
            ),
            "transposition": _keys(
                records,
                lambda record: _nested(record, "position", "transposition_key"),
            ),
        }
        for source, records in records_by_source.items()
    }
    pairs: list[dict[str, Any]] = []
    sources = sorted(groups)
    for left_index, left in enumerate(sources):
        for right in sources[left_index + 1 :]:
            pair = {"left": left, "right": right, "measured": True}
            for layer in groups[left]:
                left_keys = groups[left][layer]
                right_keys = groups[right][layer]
                intersection = left_keys & right_keys
                union = left_keys | right_keys
                pair[f"{layer}_left_count"] = len(left_keys)
                pair[f"{layer}_right_count"] = len(right_keys)
                pair[f"{layer}_overlap_count"] = len(intersection)
                pair[f"{layer}_jaccard"] = len(intersection) / len(union) if union else 0.0
                pair[f"{layer}_measurable"] = bool(left_keys or right_keys)
            pairs.append(pair)
    return {
        "schema": "phase10r_overlap_report/v1",
        "source_counts": {
            source: {layer: len(values) for layer, values in layers.items()}
            for source, layers in groups.items()
        },
        "pairs": pairs,
        "unknown_layers": [
            "canonical_sfen",
            "transposition",
        ],
        "note": (
            "An empty layer is unavailable, not evidence of no overlap. Exact raw and source "
            "identities are compared only when the adapter supplied them."
        ),
    }


def deduplicate_records(
    records_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    source_priority: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep the first record under deterministic identity priority.

    Records without an exact identity are retained but counted as unresolved;
    they are not dropped based on a guessed text similarity.
    """

    unknown_sources = sorted(set(records_by_source) - set(source_priority))
    if unknown_sources:
        raise ValueError(f"source priority is missing sources: {unknown_sources}")
    ordered: list[tuple[str, Mapping[str, Any]]] = []
    for source in source_priority:
        ordered.extend((source, record) for record in records_by_source.get(source, ()))
    seen: dict[tuple[str, str], tuple[str, str]] = {}
    kept: list[dict[str, Any]] = []
    dropped_by_source: dict[str, int] = {source: 0 for source in source_priority}
    unresolved_by_source: dict[str, int] = {source: 0 for source in source_priority}
    for source, record in ordered:
        key = _dedup_key(record)
        if key is None:
            unresolved_by_source[source] += 1
            kept.append(dict(record))
            continue
        if key in seen:
            dropped_by_source[source] += 1
            continue
        seen[key] = (source, str(record.get("record_id")))
        kept.append(dict(record))
    return kept, {
        "schema": "phase10r_deduplication_report/v1",
        "input_count": sum(len(records) for records in records_by_source.values()),
        "output_count": len(kept),
        "dropped_count": sum(dropped_by_source.values()),
        "dropped_by_source": dropped_by_source,
        "unresolved_identity_by_source": unresolved_by_source,
        "source_priority": list(source_priority),
        "identity_precedence": ["canonical_sfen", "transposition", "position", "history"],
    }


def split_name(
    record: Mapping[str, Any],
    *,
    salt: str,
    validation_basis_points: int = 1_000,
    test_basis_points: int = 1_000,
    reserved_holdout: bool = False,
) -> str:
    """Assign a stable split from exact identity, never from row order."""

    if not salt or validation_basis_points < 0 or test_basis_points < 0:
        raise ValueError("invalid deterministic split policy")
    if validation_basis_points + test_basis_points >= 10_000:
        raise ValueError("validation and test fractions must leave a training split")
    if reserved_holdout:
        return "reserved_holdout"
    key = _dedup_key(record)
    if key is None:
        return "unassigned_missing_identity"
    digest = hashlib.sha256(f"{salt}\0{key[0]}\0{key[1]}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10_000
    if bucket < test_basis_points:
        return "test"
    if bucket < test_basis_points + validation_basis_points:
        return "validation"
    return "train"


def _dedup_key(record: Mapping[str, Any]) -> tuple[str, str] | None:
    candidates = (
        ("canonical_sfen", _nested(record, "position", "canonical_sfen")),
        ("transposition", _nested(record, "position", "transposition_key")),
        ("position", _nested(record, "position", "identity")),
        ("history", _nested(record, "history", "identity")),
    )
    for namespace, value in candidates:
        if isinstance(value, Mapping):
            if value.get("exact") and value.get("digest_sha256"):
                return namespace, str(value["digest_sha256"])
        elif isinstance(value, str) and value:
            return namespace, value
    return None


def _keys(records: Sequence[Mapping[str, Any]], getter: Any) -> set[tuple[str, str]]:
    values: set[tuple[str, str]] = set()
    for record in records:
        value = getter(record)
        if isinstance(value, Mapping):
            if value.get("exact") and value.get("digest_sha256"):
                values.add((str(value.get("namespace", "unknown")), str(value["digest_sha256"])))
        elif isinstance(value, str) and value:
            values.add(("value", value))
    return values


def _nested(record: Mapping[str, Any], *keys: str) -> Any:
    value: Any = record
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


__all__ = [
    "deduplicate_records",
    "measure_phase10r_overlap",
    "split_name",
]
