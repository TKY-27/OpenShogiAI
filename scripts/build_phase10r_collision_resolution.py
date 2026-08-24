#!/usr/bin/env python3
"""Freeze every collision group from the latest Phase 10R partial scan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from open_shogi_training.data.phase10r_identity import resolve_canonical_collision


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(database: Path, source_report: Path) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{database.resolve()}?immutable=1", uri=True)
    collision_ids = [
        str(row[0])
        for row in connection.execute(
            """
            SELECT canonical_position_id FROM positions
            GROUP BY canonical_position_id
            HAVING COUNT(DISTINCT split) > 1
                OR COUNT(DISTINCT parser_version) > 1
                OR COUNT(DISTINCT normalization_version) > 1
            ORDER BY canonical_position_id
            """
        )
    ]
    decisions: list[dict[str, Any]] = []
    split_sets: Counter[str] = Counter()
    for canonical_position_id in collision_ids:
        rows = [
            {
                "canonical_position_id": canonical_position_id,
                "split": split,
                "artifact_id": artifact_id,
                "game_id": game_id,
                "record_id": record_id,
                "position_index": position_index,
                "parser_version": parser_version,
                "normalization_version": normalization_version,
                "source_id": source_id,
            }
            for (
                split,
                artifact_id,
                game_id,
                record_id,
                position_index,
                parser_version,
                normalization_version,
                source_id,
            ) in connection.execute(
                """
                SELECT split, artifact_id, game_id, record_id, position_index,
                       parser_version, normalization_version, source_id
                FROM positions WHERE canonical_position_id = ?
                ORDER BY split, artifact_id, game_id, record_id, position_index
                """,
                (canonical_position_id,),
            )
        ]
        decision = resolve_canonical_collision(rows)
        decision["observed_sources"] = sorted({str(row["source_id"]) for row in rows})
        decision["observed_artifacts"] = sorted({str(row["artifact_id"]) for row in rows})
        decision["evidence_status"] = "frozen_rule_not_applied_to_legacy_partial_database"
        decisions.append(decision)
        split_sets[",".join(decision["observed_splits"])] += 1
    connection.close()
    cross_split = sum(len(item["observed_splits"]) > 1 for item in decisions)
    final_holdout = sum(item["protected_holdout_collision"] for item in decisions)
    version = sum(item["parser_normalization_collision"] for item in decisions)
    return {
        "schema": "open_shogiai_phase10r_collision_resolution/v2",
        "frozen_at": "2026-08-24",
        "evidence_database": database.name,
        "evidence_database_sha256": _sha256_file(database),
        "source_report": source_report.name,
        "source_report_sha256": _sha256_file(source_report),
        "identity_schema": "legacy open_shogiai_phase10r_scanned_position/v1 evidence",
        "resolution_schema": "open_shogiai_phase10r_identity/v2",
        "canonical_cross_split_collision_count": cross_split,
        "protected_final_holdout_collision_count": final_holdout,
        "parser_normalization_collision_count": version,
        "union_collision_count": len(decisions),
        "split_set_counts": dict(sorted(split_sets.items())),
        "rule": (
            "retain the highest-precedence split and its deterministic owner; "
            "exclude every other occurrence without split reassignment"
        ),
        "protected_holdout_moved_to_training": 0,
        "decisions": decisions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build(args.database, args.source_report)
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
