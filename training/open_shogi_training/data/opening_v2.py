"""Build the provenance-bound OpenShogiAI opening-book v2 artifact."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from open_shogi_training.data.gzip_jsonl import (
    ArtifactDigest,
    compact_json_bytes,
    iter_jsonl_gzip,
    write_jsonl_gzip_atomic,
)

OPENING_BOOK_SCHEMA = "open_shogi_opening_book/v2"
_LEGACY_SCHEMA = "phase3_opening_export/v1"
_LABEL_SCHEMA = "phase4_teacher_label/v1"
_RULE_PROFILE = "standard-shogi/v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MOVE_RE = re.compile(r"^(?:[1-9][a-i][1-9][a-i]\+?|[RBGSNLP]\*[1-9][a-i])$")
_MAX_LABEL_BYTES = 64 * 1024 * 1024
_MAX_MANIFEST_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class OpeningV2BuildReport:
    input_records: int
    retained_positions: int
    retained_candidates: int
    rejected_without_teacher: int
    rejected_teacher_loss: int
    ibisha_candidates: int
    ibisha_vs_furibisha_candidates: int
    artifact: ArtifactDigest


@dataclass(frozen=True, slots=True)
class _TeacherEvidence:
    score_cp: int
    depth: int
    nodes: int
    scores: tuple[int, ...]
    provenance: tuple[str, ...]


def build_opening_book_v2(
    legacy_opening: Path,
    labels: Path,
    label_manifest: Path,
    dataset_manifest: Path,
    output: Path,
    *,
    build_version: str,
    maximum_plies: int = 40,
    minimum_sample_count: int = 2,
    maximum_teacher_loss_cp: int = 80,
) -> OpeningV2BuildReport:
    """Join approved game statistics to teacher MultiPV evidence and publish deterministic v2."""

    if not build_version or len(build_version.encode()) > 128 or not build_version.isascii():
        raise ValueError("build_version must be non-empty bounded ASCII")
    if not 1 <= maximum_plies <= 40:
        raise ValueError("maximum_plies must be 1..=40")
    if not 1 <= minimum_sample_count <= 1_000_000:
        raise ValueError("minimum_sample_count must be 1..=1000000")
    if not 0 <= maximum_teacher_loss_cp <= 10_000:
        raise ValueError("maximum_teacher_loss_cp must be 0..=10000")

    dataset = _load_unique_json(dataset_manifest, _MAX_MANIFEST_BYTES)
    label_index = _load_unique_json(label_manifest, _MAX_MANIFEST_BYTES)
    _validate_manifests(dataset, label_index, dataset_manifest, labels)
    dataset_sha = _sha256_file(dataset_manifest)
    label_manifest_sha = _sha256_file(label_manifest)
    labels_sha = _sha256_file(labels)
    legacy_sha = _sha256_file(legacy_opening)
    global_provenance = tuple(sorted({dataset_sha, label_manifest_sha, labels_sha, legacy_sha}))

    teachers = _load_teacher_evidence(labels, maximum_plies, global_provenance)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    input_records = 0
    rejected_without_teacher = 0
    for row in iter_jsonl_gzip(
        legacy_opening,
        max_compressed_bytes=64 * 1024 * 1024,
        max_uncompressed_bytes=256 * 1024 * 1024,
        max_records=1_000_000,
        max_line_bytes=1024 * 1024,
    ):
        input_records += 1
        _validate_legacy_row(row)
        if row["count"] < minimum_sample_count:
            continue
        teacher = teachers.get((row["stateSfen"], row["moveUsi"]))
        if teacher is None:
            rejected_without_teacher += 1
            continue
        grouped[row["stateSfen"]].append(_candidate(row, teacher))

    retained: list[dict[str, Any]] = []
    rejected_teacher_loss = 0
    ibisha_candidates = 0
    ibisha_vs_furibisha_candidates = 0
    for state_sfen, candidates in sorted(grouped.items()):
        best_score = max(candidate["teacherScoreCp"] for candidate in candidates)
        safe = [
            candidate
            for candidate in candidates
            if best_score - candidate["teacherScoreCp"] <= maximum_teacher_loss_cp
        ]
        rejected_teacher_loss += len(candidates) - len(safe)
        if not safe:
            continue
        safe.sort(
            key=lambda candidate: (
                -candidate["teacherScoreCp"],
                -candidate["sampleCount"],
                candidate["moveUsi"],
            )
        )
        for candidate in safe:
            classification = _classify_opening(state_sfen, candidate["moveUsi"])
            candidate["openingClassification"] = classification
            ibisha_candidates += int(classification == "ibisha")
            ibisha_vs_furibisha_candidates += int(classification == "ibisha-vs-furibisha")
        state_key = hashlib.sha256(state_sfen.encode()).hexdigest()
        record_provenance = sorted(
            set(global_provenance).union(*(candidate["provenanceReferences"] for candidate in safe))
        )
        record: dict[str, Any] = {
            "schema": OPENING_BOOK_SCHEMA,
            "stateKey": state_key,
            "stateSfen": state_sfen,
            "ruleProfile": _RULE_PROFILE,
            "buildVersion": build_version,
            "provenanceReferences": record_provenance,
            "candidates": safe,
        }
        record["recordChecksum"] = hashlib.sha256(compact_json_bytes(record)).hexdigest()
        retained.append(record)

    if not retained:
        raise ValueError("no teacher-safe opening candidates survived")
    artifact = write_jsonl_gzip_atomic(output, retained)
    return OpeningV2BuildReport(
        input_records=input_records,
        retained_positions=len(retained),
        retained_candidates=sum(len(record["candidates"]) for record in retained),
        rejected_without_teacher=rejected_without_teacher,
        rejected_teacher_loss=rejected_teacher_loss,
        ibisha_candidates=ibisha_candidates,
        ibisha_vs_furibisha_candidates=ibisha_vs_furibisha_candidates,
        artifact=artifact,
    )


def _load_teacher_evidence(
    path: Path,
    maximum_plies: int,
    global_provenance: tuple[str, ...],
) -> dict[tuple[str, str], _TeacherEvidence]:
    if path.stat().st_size > _MAX_LABEL_BYTES:
        raise ValueError(f"labels exceed {_MAX_LABEL_BYTES} bytes")
    evidence: dict[tuple[str, str], list[tuple[int, int, int, tuple[str, ...]]]] = defaultdict(list)
    with path.open("rb") as label_file:
        for number, raw in enumerate(label_file, start=1):
            if len(raw) > 1024 * 1024:
                raise ValueError(f"teacher label {number} exceeds the line limit")
            row = _loads_unique(raw, f"teacher label {number}")
            if row.get("schema") != _LABEL_SCHEMA:
                raise ValueError(f"teacher label {number} has an unsupported schema")
            position_index = row.get("position_index")
            if not isinstance(position_index, int) or isinstance(position_index, bool):
                raise ValueError(f"teacher label {number} has invalid position_index")
            if position_index >= maximum_plies:
                continue
            canonical_sfen = row.get("canonical_sfen")
            if not isinstance(canonical_sfen, str):
                raise ValueError(f"teacher label {number} lacks canonical_sfen")
            state_sfen = _state_sfen(canonical_sfen)
            provenance = set(global_provenance)
            for value in (
                row.get("position_id"),
                row.get("config_sha256"),
                row.get("dataset_manifest_sha256"),
                row.get("teacher", {}).get("binary_sha256"),
            ):
                if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                    raise ValueError(f"teacher label {number} has invalid provenance")
                provenance.add(value)
            candidates = row.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                raise ValueError(f"teacher label {number} lacks candidates")
            for candidate in candidates:
                movement = candidate.get("pv", [None])[0]
                score = candidate.get("score", {})
                score_value = score.get("value")
                if score.get("kind") == "mate" and isinstance(score_value, int):
                    score_value = 32_000 if score_value > 0 else -32_000
                depth = candidate.get("depth")
                nodes = candidate.get("nodes")
                if (
                    not isinstance(movement, str)
                    or _MOVE_RE.fullmatch(movement) is None
                    or score.get("kind") not in {"cp", "mate"}
                    or not isinstance(score_value, int)
                    or isinstance(score_value, bool)
                    or not isinstance(depth, int)
                    or isinstance(depth, bool)
                    or not 1 <= depth <= 64
                    or not isinstance(nodes, int)
                    or isinstance(nodes, bool)
                    or nodes <= 0
                ):
                    raise ValueError(f"teacher label {number} has invalid candidate evidence")
                evidence[(state_sfen, movement)].append(
                    (score_value, depth, nodes, tuple(sorted(provenance)))
                )
    result: dict[tuple[str, str], _TeacherEvidence] = {}
    for key, observations in evidence.items():
        selected = max(observations, key=lambda item: (item[2], item[1], -abs(item[0])))
        result[key] = _TeacherEvidence(
            score_cp=selected[0],
            depth=selected[1],
            nodes=selected[2],
            scores=tuple(observation[0] for observation in observations),
            provenance=selected[3],
        )
    return result


def _candidate(row: dict[str, Any], teacher: _TeacherEvidence) -> dict[str, Any]:
    side = row["stateSfen"].split(" ")[1]
    black = _empty_results()
    white = _empty_results()
    target = black if side == "b" else white
    target.update(
        {
            "wins": row["wins"],
            "losses": row["losses"],
            "draws": row["draws"],
            "unknown": row["unknown"],
        }
    )
    uncertainty = max(teacher.scores) - min(teacher.scores) if len(teacher.scores) > 1 else None
    return {
        "moveUsi": row["moveUsi"],
        "sampleCount": row["count"],
        "sourceDistribution": row["sourceCounts"],
        "blackResults": black,
        "whiteResults": white,
        "teacherScoreCp": teacher.score_cp,
        "scoreUncertaintyCp": uncertainty,
        "teacherDepth": teacher.depth,
        "teacherNodes": teacher.nodes,
        "openingClassification": "unclassified",
        "provenanceReferences": list(teacher.provenance),
    }


def _classify_opening(state_sfen: str, movement: str) -> str:
    """Classify using rook location, rook-pawn pressure, king direction, and candidate structure."""

    board, side, _hands = state_sfen.split(" ")
    pieces = _parse_board(board)
    black_rooks = [file for (file, _rank), piece in pieces.items() if piece in {"R", "+R"}]
    white_rooks = [file for (file, _rank), piece in pieces.items() if piece in {"r", "+r"}]
    own_rooks = black_rooks if side == "b" else white_rooks
    enemy_rooks = white_rooks if side == "b" else black_rooks
    home_file = 2 if side == "b" else 8
    enemy_home_file = 8 if side == "b" else 2
    own_king = next(
        (file for (file, _rank), piece in pieces.items() if piece == ("K" if side == "b" else "k")),
        5,
    )
    own_furibisha = any(file != home_file for file in own_rooks)
    enemy_furibisha = any(file != enemy_home_file for file in enemy_rooks)

    from_file = int(movement[0]) if "*" not in movement else None
    to_file = int(movement[2])
    from_rank = movement[1] if "*" not in movement else None
    direct_rook_pawn = (side == "b" and from_file == 2 and from_rank in {"g", "f"}) or (
        side == "w" and from_file == 8 and from_rank in {"c", "d"}
    )
    moving_piece = pieces.get((from_file, ord(from_rank) - ord("a") + 1)) if from_file else None
    candidate_lateral_rook = moving_piece in {"R", "r", "+R", "+r"} and to_file != home_file
    own_furibisha = own_furibisha or candidate_lateral_rook

    rook_pawn_advanced = (home_file, 7 if side == "b" else 3) not in pieces
    king_opposite_rook = own_king >= 6 if side == "b" else own_king <= 4
    ibisha_signals = int(direct_rook_pawn) + int(rook_pawn_advanced) + int(king_opposite_rook)
    if own_furibisha:
        return "furibisha"
    if ibisha_signals >= 1:
        return "ibisha-vs-furibisha" if enemy_furibisha else "ibisha"
    return "unclassified"


def _parse_board(board: str) -> dict[tuple[int, int], str]:
    pieces: dict[tuple[int, int], str] = {}
    ranks = board.split("/")
    if len(ranks) != 9:
        raise ValueError("opening SFEN board must contain nine ranks")
    for rank, encoded in enumerate(ranks, start=1):
        file = 9
        index = 0
        while index < len(encoded):
            token = encoded[index]
            if token.isdigit():
                file -= int(token)
                index += 1
                continue
            if token == "+":
                index += 1
                if index >= len(encoded):
                    raise ValueError("opening SFEN has truncated promotion")
                token += encoded[index]
            if not 1 <= file <= 9:
                raise ValueError("opening SFEN rank width is invalid")
            pieces[(file, rank)] = token
            file -= 1
            index += 1
        if file != 0:
            raise ValueError("opening SFEN rank width is invalid")
    return pieces


def _validate_manifests(
    dataset: dict[str, Any],
    labels: dict[str, Any],
    dataset_path: Path,
    labels_path: Path,
) -> None:
    if dataset.get("schema") != "phase3_dataset_manifest/v1":
        raise ValueError("dataset manifest schema is unsupported")
    source = dataset.get("source")
    if not isinstance(source, dict) or not all(
        (
            source.get("sourceId") == "aobazero-no-noise",
            source.get("license") == "Public Domain",
            source.get("redistributable") is True,
            source.get("machineLearningAllowed") is True,
        )
    ):
        raise ValueError("dataset source is not approved for this opening build")
    if labels.get("schema") != "phase4_teacher_label_manifest/v2":
        raise ValueError("label manifest schema is unsupported")
    expected_dataset = labels.get("dataset_manifest")
    expected_labels = labels.get("artifacts", {}).get("labels.jsonl")
    if (
        not isinstance(expected_dataset, dict)
        or expected_dataset.get("sha256") != _sha256_file(dataset_path)
        or not isinstance(expected_labels, dict)
        or expected_labels.get("sha256") != _sha256_file(labels_path)
        or expected_labels.get("size") != labels_path.stat().st_size
    ):
        raise ValueError("teacher-label manifest artifact binding mismatch")


def _validate_legacy_row(row: dict[str, Any]) -> None:
    if row.get("schema") != _LEGACY_SCHEMA:
        raise ValueError("legacy opening row schema is unsupported")
    for key in ("stateKey", "stateSfen", "moveUsi", "count", "sourceCounts"):
        if key not in row:
            raise ValueError(f"legacy opening row lacks {key}")
    if (
        _SHA256_RE.fullmatch(row["stateKey"]) is None
        or hashlib.sha256(row["stateSfen"].encode()).hexdigest() != row["stateKey"]
        or _MOVE_RE.fullmatch(row["moveUsi"]) is None
        or not isinstance(row["count"], int)
        or isinstance(row["count"], bool)
        or row["count"] <= 0
        or not isinstance(row["sourceCounts"], dict)
        or sum(row["sourceCounts"].values()) != row["count"]
    ):
        raise ValueError("legacy opening row identity/count is invalid")


def _state_sfen(full_sfen: str) -> str:
    fields = full_sfen.split(" ")
    if len(fields) != 4 or not fields[3].isdecimal() or int(fields[3]) <= 0:
        raise ValueError("teacher canonical_sfen is invalid")
    return " ".join(fields[:3])


def _empty_results() -> dict[str, int]:
    return {"wins": 0, "losses": 0, "draws": 0, "unknown": 0}


def _loads_unique(raw: bytes, context: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{context} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=unique)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"invalid {context}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def _load_unique_json(path: Path, maximum: int) -> dict[str, Any]:
    if path.stat().st_size > maximum:
        raise ValueError(f"{path} exceeds {maximum} bytes")
    return _loads_unique(path.read_bytes(), str(path))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
