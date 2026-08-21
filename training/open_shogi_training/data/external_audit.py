"""Bounded readers and source-preserving normalization for Phase 10A.

The readers in this module are audit utilities, not training importers.  They
accept bounded samples, retain source-specific labels, and deliberately avoid
replaying positions or deciding what an evaluation field means.  In
particular, an adapter never turns an unknown score perspective into a common
training target.
"""

from __future__ import annotations

import hashlib
import re
import struct
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from open_shogi_training.data.aobazero import adapt_aobazero_csa

NORMALIZED_SCHEMA = "phase10a_normalized_record/v1"
HCPE_RECORD_SIZE = 38
HCPE3_HEADER_SIZE = 36
HCPE3_MOVE_INFO_SIZE = 6
HCPE3_VISIT_SIZE = 4
PACKED_SFEN_VALUE_SIZE = 40
DEFAULT_MAX_SAMPLE_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_RECORDS = 100_000
DEFAULT_MAX_HCPE3_MOVES = 512
DEFAULT_MAX_HCPE3_CANDIDATES = 512

_CSA_MOVE_RE = re.compile(r"^(?P<move>[+-][0-9]{4}[A-Z]{2})(?P<annotation>.*)$")
_KIF_MOVE_RE = re.compile(r"^\s*(?P<number>[0-9]+)\s+(?P<move>\S+)(?:\s+(?P<rest>.*))?$")
_KIF_EVAL_RE = re.compile(r"評価値\s+(?P<value>[-+]?\d+)(?:\s+読み筋\s+(?P<pv>.*))?")
_SCORE_RE = re.compile(r"(?:^|[,'])v=(?P<value>[^,']+)")
_RESULT_NAMES = {0: "draw", 1: "black_win", 2: "white_win"}


class ExternalAuditFormatError(ValueError):
    """Raised when a bounded sample is malformed or exceeds a reader limit."""


def parse_csa_sample(
    raw: bytes,
    *,
    source_id: str,
    artifact_id: str,
    license_decision: Mapping[str, Any] | None = None,
    strict_aobazero: bool = False,
    max_bytes: int = DEFAULT_MAX_SAMPLE_BYTES,
    max_records: int = DEFAULT_MAX_RECORDS,
) -> list[dict[str, Any]]:
    """Read CSA move prefixes without claiming legal position reconstruction."""

    _check_bounds(raw, max_bytes=max_bytes)
    if strict_aobazero:
        adapt_aobazero_csa(raw)
    text, encoding = _decode_text(raw, format_name="CSA")

    records: list[dict[str, Any]] = []
    for game_index, game_lines in enumerate(_split_csa_games(text.splitlines())):
        move_lines = [line for line in game_lines if _CSA_MOVE_RE.match(line)]
        if not move_lines:
            raise ExternalAuditFormatError(f"CSA record {game_index} has no moves")
        terminal = next((line for line in reversed(game_lines) if line.startswith("%")), None)
        prefix: list[str] = []
        for ply, line in enumerate(move_lines, start=1):
            match = _CSA_MOVE_RE.match(line)
            assert match is not None
            prefix.append(match.group("move"))
            position_digest = _sha256_text("csa-move-prefix\n" + "\n".join(prefix))
            annotation = match.group("annotation")
            raw_score = _raw_score(annotation)
            records.append(
                _normalized_record(
                    source_id=source_id,
                    artifact_id=artifact_id,
                    record_id=f"game-{game_index:04d}-ply-{ply:04d}",
                    format_name="csa",
                    position_identity={
                        "namespace": "csa_move_prefix",
                        "digest_sha256": position_digest,
                        "exact": False,
                        "identity_note": "CSA text was not replayed by a board parser.",
                    },
                    history_identity={
                        "namespace": "csa_move_prefix",
                        "digest_sha256": position_digest,
                        "exact": False,
                    },
                    labels={
                        "played_move": match.group("move"),
                        "best_move": None,
                        "policy_distribution": None,
                        "wdl": {"raw_terminal": terminal, "normalized": None},
                        "result": terminal,
                        "raw_source_score": raw_score,
                        "source_score_semantics": (
                            "AobaZero/CSA annotation v field retained without interpretation"
                            if raw_score is not None
                            else None
                        ),
                        "score_perspective": "unknown",
                        "mate_representation": "raw CSA terminal retained",
                        "raw_annotation": annotation or None,
                    },
                    search={"nodes": None, "playouts": None, "depth": None},
                    raw_record_sha256=_sha256(raw),
                    license_decision=license_decision,
                    text_encoding=encoding,
                )
            )
            _check_record_limit(records, max_records)
    return records


def parse_kif_sample(
    raw: bytes,
    *,
    source_id: str,
    artifact_id: str,
    license_decision: Mapping[str, Any] | None = None,
    max_bytes: int = DEFAULT_MAX_SAMPLE_BYTES,
    max_records: int = DEFAULT_MAX_RECORDS,
) -> list[dict[str, Any]]:
    """Read numbered KIF move lines while preserving Japanese move text.

    Official CSA/WCSC releases are commonly UTF-8, while Denryu archives use
    CP932.  The selected encoding is retained in each record; no transcoding
    is treated as a rights or legality decision.
    """

    _check_bounds(raw, max_bytes=max_bytes)
    text, encoding = _decode_text(raw, format_name="KIF")
    lines = text.splitlines()
    move_lines = []
    for line in lines:
        match = _KIF_MOVE_RE.match(line)
        if match is None:
            continue
        token = _kif_move_token(match.group("move"), match.group("rest"))
        if any(
            marker in token for marker in ("投了", "詰み", "持将棋", "中断", "千日手", "入玉宣言")
        ):
            continue
        move_lines.append(match)
    if not move_lines:
        raise ExternalAuditFormatError("KIF sample has no numbered moves")
    evaluations: dict[int, dict[str, Any]] = {}
    current_number: int | None = None
    for line in lines:
        move_match = _KIF_MOVE_RE.match(line)
        if move_match:
            current_number = int(move_match.group("number"))
            continue
        if current_number is None or not line.lstrip().startswith("**"):
            continue
        evaluation = _KIF_EVAL_RE.search(line)
        if evaluation:
            evaluations[current_number] = {
                "value": int(evaluation.group("value")),
                "pv": evaluation.group("pv"),
                "raw": line.strip(),
            }

    records: list[dict[str, Any]] = []
    prefix: list[str] = []
    for index, match in enumerate(move_lines, start=1):
        move = _kif_move_token(match.group("move"), match.group("rest"))
        move_number = int(match.group("number"))
        evaluation = evaluations.get(move_number)
        prefix.append(move)
        digest = _sha256_text("kif-move-prefix\n" + "\n".join(prefix))
        records.append(
            _normalized_record(
                source_id=source_id,
                artifact_id=artifact_id,
                record_id=f"game-0000-ply-{index:04d}",
                format_name="kif",
                position_identity={
                    "namespace": "kif_move_prefix",
                    "digest_sha256": digest,
                    "exact": False,
                    "identity_note": "KIF text was not replayed by a board parser.",
                },
                history_identity={
                    "namespace": "kif_move_prefix",
                    "digest_sha256": digest,
                    "exact": False,
                },
                labels={
                    "played_move": move,
                    "best_move": None,
                    "policy_distribution": None,
                    "wdl": {"raw": None, "normalized": None},
                    "result": None,
                    "raw_source_score": evaluation["value"] if evaluation else None,
                    "source_score_semantics": (
                        "KIF 対局 評価値 integer; unit and perspective retained"
                        if evaluation
                        else None
                    ),
                    "score_perspective": "unknown",
                    "mate_representation": "not present in bounded move line",
                    "raw_move_suffix": match.group("rest"),
                    "raw_evaluation": evaluation,
                },
                search={"nodes": None, "playouts": None, "depth": None},
                raw_record_sha256=_sha256(raw),
                license_decision=license_decision,
                text_encoding=encoding,
            )
        )
        _check_record_limit(records, max_records)
    return records


def parse_hcpe_sample(
    raw: bytes,
    *,
    source_id: str,
    artifact_id: str,
    license_decision: Mapping[str, Any] | None = None,
    max_bytes: int = DEFAULT_MAX_SAMPLE_BYTES,
    max_records: int = DEFAULT_MAX_RECORDS,
) -> list[dict[str, Any]]:
    """Read fixed-size 38-byte HCPE records using the published layout."""

    _check_bounds(raw, max_bytes=max_bytes)
    if len(raw) % HCPE_RECORD_SIZE:
        raise ExternalAuditFormatError("HCPE sample is not a multiple of 38 bytes")

    records: list[dict[str, Any]] = []
    for index, (hcp, score, best_move, result, _dummy) in enumerate(
        struct.iter_unpack("<32shhbB", raw)
    ):
        result_name = _RESULT_NAMES.get(result)
        position_digest = hashlib.sha256(hcp).hexdigest()
        records.append(
            _normalized_record(
                source_id=source_id,
                artifact_id=artifact_id,
                record_id=f"record-{index:08d}",
                format_name="hcpe",
                position_identity={
                    "namespace": "hcp",
                    "digest_sha256": position_digest,
                    "exact": True,
                    "encoding": "32-byte HuffmanCodedPos",
                },
                history_identity={
                    "namespace": "hcpe_history",
                    "digest_sha256": None,
                    "exact": False,
                    "availability": "not_present_in_hcpe_record",
                },
                labels={
                    "played_move": None,
                    "best_move": best_move,
                    "policy_distribution": None,
                    "wdl": {"raw_code": result, "normalized": result_name},
                    "result": result_name,
                    "raw_source_score": score,
                    "source_score_semantics": (
                        "HCPE eval field; unit and perspective retained as unknown"
                    ),
                    "score_perspective": "unknown",
                    "mate_representation": "not separately encoded; raw eval retained",
                },
                search={"nodes": None, "playouts": None, "depth": None},
                raw_record_sha256=_sha256(
                    raw[index * HCPE_RECORD_SIZE : (index + 1) * HCPE_RECORD_SIZE]
                ),
                license_decision=license_decision,
            )
        )
        _check_record_limit(records, max_records)
    return records


def parse_hcpe3_sample(
    raw: bytes,
    *,
    source_id: str,
    artifact_id: str,
    license_decision: Mapping[str, Any] | None = None,
    max_bytes: int = DEFAULT_MAX_SAMPLE_BYTES,
    max_records: int = DEFAULT_MAX_RECORDS,
    max_moves: int = DEFAULT_MAX_HCPE3_MOVES,
    max_candidates: int = DEFAULT_MAX_HCPE3_CANDIDATES,
) -> list[dict[str, Any]]:
    """Read variable-length HCPE3 games and retain candidate visit counts."""

    _check_bounds(raw, max_bytes=max_bytes)
    records: list[dict[str, Any]] = []
    offset = 0
    game_index = 0
    while offset < len(raw):
        if len(raw) - offset < HCPE3_HEADER_SIZE:
            raise ExternalAuditFormatError("truncated HCPE3 header")
        hcp, move_num, result, game_info = struct.unpack_from("<32sHBB", raw, offset)
        offset += HCPE3_HEADER_SIZE
        if move_num > max_moves:
            raise ExternalAuditFormatError(f"HCPE3 move count exceeds {max_moves}")
        for ply in range(move_num):
            if len(raw) - offset < HCPE3_MOVE_INFO_SIZE:
                raise ExternalAuditFormatError("truncated HCPE3 move info")
            selected_move, score, candidate_num = struct.unpack_from("<hhH", raw, offset)
            offset += HCPE3_MOVE_INFO_SIZE
            if candidate_num > max_candidates:
                raise ExternalAuditFormatError(f"HCPE3 candidate count exceeds {max_candidates}")
            visits: list[dict[str, Any]] = []
            for _candidate_index in range(candidate_num):
                if len(raw) - offset < HCPE3_VISIT_SIZE:
                    raise ExternalAuditFormatError("truncated HCPE3 candidate visit")
                move16, visit_num = struct.unpack_from("<hH", raw, offset)
                offset += HCPE3_VISIT_SIZE
                visits.append({"move16": move16, "visits": visit_num})
            total_visits = sum(item["visits"] for item in visits)
            for item in visits:
                item["probability"] = item["visits"] / total_visits if total_visits else 0.0
            prefix_digest = _sha256_bytes(b"hcpe3-prefix\0" + hcp + struct.pack("<H", ply))
            records.append(
                _normalized_record(
                    source_id=source_id,
                    artifact_id=artifact_id,
                    record_id=f"game-{game_index:04d}-ply-{ply:04d}",
                    format_name="hcpe3",
                    position_identity={
                        "namespace": "hcpe3_prefix",
                        "digest_sha256": prefix_digest,
                        "exact": False,
                        "start_position_hcp_sha256": hashlib.sha256(hcp).hexdigest(),
                        "identity_note": (
                            "HCPE3 move prefix was not replayed; the 32-byte HCP start "
                            "position is retained separately."
                        ),
                    },
                    history_identity={
                        "namespace": "hcpe3_prefix",
                        "digest_sha256": prefix_digest,
                        "exact": False,
                    },
                    labels={
                        "played_move": None,
                        "best_move": selected_move,
                        "policy_distribution": visits,
                        "wdl": {
                            "raw_result_code": result,
                            "normalized": _RESULT_NAMES.get(result & 0x03),
                        },
                        "result": _RESULT_NAMES.get(result & 0x03),
                        "raw_source_score": score,
                        "source_score_semantics": (
                            "HCPE3 eval field; unit and perspective retained as unknown"
                        ),
                        "score_perspective": "unknown",
                        "mate_representation": "result/game-info flags retained as raw values",
                        "raw_game_info": game_info,
                    },
                    search={
                        "nodes": None,
                        "playouts": total_visits,
                        "depth": None,
                    },
                    raw_record_sha256=_sha256(raw),
                    license_decision=license_decision,
                )
            )
            _check_record_limit(records, max_records)
        game_index += 1
    if not records:
        raise ExternalAuditFormatError("HCPE3 sample has no moves")
    return records


def parse_packed_sfen_value_sample(
    raw: bytes,
    *,
    source_id: str,
    artifact_id: str,
    format_name: str = "packed_sfen_value",
    license_decision: Mapping[str, Any] | None = None,
    max_bytes: int = DEFAULT_MAX_SAMPLE_BYTES,
    max_records: int = DEFAULT_MAX_RECORDS,
) -> list[dict[str, Any]]:
    """Read 40-byte YaneuraOu/cshogi PackedSfenValue records."""

    _check_bounds(raw, max_bytes=max_bytes)
    if len(raw) % PACKED_SFEN_VALUE_SIZE:
        raise ExternalAuditFormatError("PackedSfenValue sample is not a multiple of 40 bytes")

    records: list[dict[str, Any]] = []
    for index, (sfen, score, best_move, game_ply, game_result, _padding) in enumerate(
        struct.iter_unpack("<32shHHbB", raw)
    ):
        position_digest = hashlib.sha256(sfen).hexdigest()
        records.append(
            _normalized_record(
                source_id=source_id,
                artifact_id=artifact_id,
                record_id=f"record-{index:08d}",
                format_name=format_name,
                position_identity={
                    "namespace": "packed_sfen",
                    "digest_sha256": position_digest,
                    "exact": True,
                    "encoding": "32-byte PackedSfen",
                },
                history_identity={
                    "namespace": "packed_sfen_history",
                    "digest_sha256": None,
                    "exact": False,
                    "availability": "game history is not encoded in one PSV record",
                },
                labels={
                    "played_move": None,
                    "best_move": best_move,
                    "policy_distribution": None,
                    "wdl": {"raw_code": game_result, "normalized": None},
                    "result": None,
                    "raw_source_score": score,
                    "source_score_semantics": (
                        "PackedSfenValue score field; unit and perspective retained as unknown"
                    ),
                    "score_perspective": "unknown",
                    "mate_representation": "not separately encoded; raw score retained",
                    "game_ply": game_ply,
                },
                search={"nodes": None, "playouts": None, "depth": None},
                raw_record_sha256=_sha256(
                    raw[index * PACKED_SFEN_VALUE_SIZE : (index + 1) * PACKED_SFEN_VALUE_SIZE]
                ),
                license_decision=license_decision,
            )
        )
        _check_record_limit(records, max_records)
    return records


def sample_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Produce a compact summary suitable for the machine-readable handoff."""

    formats = sorted({str(record["source_artifact"]["format"]) for record in records})
    exact_positions = sum(bool(record["position_identity"].get("exact")) for record in records)
    policy_rows = sum(record["labels"].get("policy_distribution") is not None for record in records)
    return {
        "schema": NORMALIZED_SCHEMA,
        "record_count": len(records),
        "formats": formats,
        "exact_position_identity_count": exact_positions,
        "policy_distribution_record_count": policy_rows,
        "source_ids": sorted({str(record["source_artifact"]["source_id"]) for record in records}),
    }


def measure_exact_overlap(
    records_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Measure only exact identities; incompatible namespaces remain unknown."""

    position_keys = _identity_sets_from_records(records_by_source, "position_identity")
    history_keys = _identity_sets_from_records(records_by_source, "history_identity")

    pairs: list[dict[str, Any]] = []
    source_ids = sorted(position_keys)
    for left_index, left in enumerate(source_ids):
        for right in source_ids[left_index + 1 :]:
            position_intersection = position_keys[left] & position_keys[right]
            position_union = position_keys[left] | position_keys[right]
            history_intersection = history_keys[left] & history_keys[right]
            history_union = history_keys[left] | history_keys[right]
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "left_exact_position_count": len(position_keys[left]),
                    "right_exact_position_count": len(position_keys[right]),
                    "exact_position_overlap_count": len(position_intersection),
                    "exact_position_jaccard": (
                        len(position_intersection) / len(position_union) if position_union else 0.0
                    ),
                    "left_exact_game_count": len(history_keys[left]),
                    "right_exact_game_count": len(history_keys[right]),
                    "exact_game_overlap_count": len(history_intersection),
                    "exact_game_jaccard": (
                        len(history_intersection) / len(history_union) if history_union else 0.0
                    ),
                    "game_identity_measurable": bool(history_union),
                    "measured": True,
                    "note": (
                        "Only exact position/history namespaces and digests are compared; "
                        "an empty history set means game overlap is unavailable."
                    ),
                }
            )
    return {
        "source_exact_position_counts": {key: len(value) for key, value in position_keys.items()},
        "source_exact_game_counts": {key: len(value) for key, value in history_keys.items()},
        "pairs": pairs,
    }


def measure_split_contamination(
    records_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Find exact-position and exact-history leakage across named splits."""

    position_keys = _identity_sets(records_by_split, "position_identity")
    history_keys = _identity_sets(records_by_split, "history_identity")
    position_pairs = _pairwise_intersections(position_keys)
    history_pairs = _pairwise_intersections(history_keys)
    return {
        "position_overlap_pairs": position_pairs,
        "history_overlap_pairs": history_pairs,
        "measured_only_for_exact_identity": True,
    }


def _normalized_record(
    *,
    source_id: str,
    artifact_id: str,
    record_id: str,
    format_name: str,
    position_identity: Mapping[str, Any],
    history_identity: Mapping[str, Any],
    labels: Mapping[str, Any],
    search: Mapping[str, Any],
    raw_record_sha256: str,
    license_decision: Mapping[str, Any] | None,
    text_encoding: str | None = None,
) -> dict[str, Any]:
    source_artifact = {
        "source_id": source_id,
        "artifact_id": artifact_id,
        "format": format_name,
    }
    if text_encoding is not None:
        source_artifact["encoding"] = text_encoding
    return {
        "schema": NORMALIZED_SCHEMA,
        "record_id": record_id,
        "source_artifact": source_artifact,
        "position_identity": dict(position_identity),
        "history_identity": dict(history_identity),
        "labels": dict(labels),
        "search": dict(search),
        "provenance": {
            "raw_record_sha256": raw_record_sha256,
            "retrieval_scope": "bounded sample only",
            "parser": "open_shogi_training.data.external_audit",
        },
        "license_decision": dict(license_decision or {}),
    }


def _split_csa_games(lines: Iterable[str]) -> list[list[str]]:
    games: list[list[str]] = [[]]
    for line in lines:
        if line.strip() == "/":
            if games[-1]:
                games.append([])
            continue
        games[-1].append(line)
    return [game for game in games if game]


def _identity_sets(
    records_by_split: Mapping[str, Sequence[Mapping[str, Any]]], field: str
) -> dict[str, set[tuple[str, str]]]:
    return _identity_sets_from_records(records_by_split, field)


def _identity_sets_from_records(
    records_by_group: Mapping[str, Sequence[Mapping[str, Any]]], field: str
) -> dict[str, set[tuple[str, str]]]:
    return {
        group: {
            (str(record[field].get("namespace")), str(record[field].get("digest_sha256")))
            for record in records
            if record[field].get("exact") and record[field].get("digest_sha256")
        }
        for group, records in records_by_group.items()
    }


def _pairwise_intersections(
    identity_sets: Mapping[str, set[tuple[str, str]]],
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    split_names = sorted(identity_sets)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "overlap_count": len(identity_sets[left] & identity_sets[right]),
                    "measured": True,
                }
            )
    return pairs


def _raw_score(annotation: str) -> float | str | None:
    match = _SCORE_RE.search(annotation)
    if not match:
        return None
    value = match.group("value")
    try:
        return float(value)
    except ValueError:
        return value


def _check_bounds(raw: bytes, *, max_bytes: int) -> None:
    if len(raw) > max_bytes:
        raise ExternalAuditFormatError(f"sample exceeds {max_bytes} bytes")


def _check_record_limit(records: Sequence[Any], max_records: int) -> None:
    if len(records) > max_records:
        raise ExternalAuditFormatError(f"sample exceeds {max_records} normalized records")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_bytes(raw: bytes) -> str:
    return _sha256(raw)


def _sha256_text(text: str) -> str:
    return _sha256(text.encode("utf-8"))


def _decode_text(raw: bytes, *, format_name: str) -> tuple[str, str]:
    for encoding in ("utf-8", "cp932"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise ExternalAuditFormatError(f"{format_name} sample is neither UTF-8 nor CP932")


def _kif_move_token(first: str, rest: str | None) -> str:
    """Keep full Japanese KIF notation, including ``同 銀(88)`` moves."""

    candidate = " ".join(part for part in (first, rest or "") if part).strip()
    match = re.match(r"^(?P<token>.*?(?:\([0-9]{2}\)|打))(?:\s+\(|$)", candidate)
    return (match.group("token") if match else first).strip()


__all__ = [
    "DEFAULT_MAX_HCPE3_CANDIDATES",
    "DEFAULT_MAX_HCPE3_MOVES",
    "DEFAULT_MAX_RECORDS",
    "DEFAULT_MAX_SAMPLE_BYTES",
    "HCPE3_HEADER_SIZE",
    "HCPE3_MOVE_INFO_SIZE",
    "HCPE3_VISIT_SIZE",
    "HCPE_RECORD_SIZE",
    "NORMALIZED_SCHEMA",
    "PACKED_SFEN_VALUE_SIZE",
    "ExternalAuditFormatError",
    "measure_exact_overlap",
    "measure_split_contamination",
    "parse_csa_sample",
    "parse_hcpe3_sample",
    "parse_hcpe_sample",
    "parse_kif_sample",
    "parse_packed_sfen_value_sample",
    "sample_summary",
]
