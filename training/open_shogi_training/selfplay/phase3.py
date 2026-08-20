"""Closed Phase 3 provenance and game/position-chain validation.

This module validates the durable Python producer contract without reimplementing
shogi rules.  Callers may additionally supply the receipt-bound Rust export rows;
when supplied, raw CSA canonicalization and every move successor are compared to
the independent engine output.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final
from urllib.parse import urlsplit

from open_shogi_training.data.registry import RegistryError, load_source_registry
from open_shogi_training.data.splits import SplitPolicy, assign_game_split
from open_shogi_training.models.dataset import validate_production_dataset_manifest

from .common import (
    ArtifactRef,
    ContractError,
    canonical_json_bytes,
    require_bool,
    require_enum,
    require_exact_keys,
    require_int,
    require_list,
    require_mapping,
    require_optional_string,
    require_sha256,
    require_string,
    validate_utc_timestamp,
    verified_artifact_descriptor,
)

_MAX_GAMES: Final = 100
_MAX_POSITIONS: Final = 100 * (2_048 + 1)
_MAX_COMPRESSED_BYTES: Final = 4 * 1024 * 1024 * 1024
_MAX_UNCOMPRESSED_BYTES: Final = 4 * 1024 * 1024 * 1024
_MAX_GAME_LINE_BYTES: Final = 32 * 1024 * 1024
_MAX_POSITION_LINE_BYTES: Final = 64 * 1024

_GAME_KEYS: Final = frozenset(
    {
        "schema",
        "gameId",
        "canonicalSha256",
        "rawObject",
        "rawCsa",
        "normalizedCsa",
        "initialSfen",
        "usiMoves",
        "plyCount",
        "positionCount",
        "outcome",
        "terminalReason",
        "resultValidation",
        "players",
        "date",
        "sourceDateTime",
        "sourceTimeZone",
        "split",
        "flags",
        "sourceId",
        "url",
        "retrievedAt",
        "licenseDecision",
    }
)
_RAW_KEYS: Final = frozenset(
    {"objectId", "objectPath", "originalFilename", "sha256", "size", "response"}
)
_RAW_RESPONSE_KEYS: Final = frozenset({"contentType", "etag", "lastModified"})
_PLAYER_KEYS: Final = frozenset({"name", "rating"})
_PLAYERS_KEYS: Final = frozenset({"black", "white"})
_FLAGS_KEYS: Final = frozenset({"short", "long"})
_LICENSE_DECISION_KEYS: Final = frozenset(
    {
        "license",
        "evidence",
        "evidenceSnapshots",
        "redistributable",
        "machineLearningAllowed",
    }
)
_POSITION_KEYS: Final = frozenset(
    {
        "schema",
        "gameId",
        "canonicalSha256",
        "rawSha256",
        "sourceId",
        "split",
        "positionIndex",
        "sfen",
        "moveUsi",
        "nextSfen",
        "outcome",
        "terminalReason",
        "sideToMove",
        "fullPlies",
        "remainingPlies",
        "eligible",
        "terminalTail",
    }
)
_EXPORT_KEYS: Final = frozenset(
    {
        "schema",
        "status",
        "inputFile",
        "normalizedCsa",
        "initialSfen",
        "positionSfens",
        "usiMoves",
        "blackName",
        "whiteName",
        "terminalReason",
        "outcome",
        "resultValidation",
    }
)


@dataclass(frozen=True, slots=True)
class ValidatedPhase3Artifacts:
    """Exact canonical Phase 3 rows retained after the full provenance check."""

    games: tuple[Mapping[str, Any], ...]
    positions: tuple[Mapping[str, Any], ...]


def validate_phase3_artifact_chain(
    *,
    repository_root: Path,
    manifest: Mapping[str, Any],
    dataset_manifest_ref: ArtifactRef,
    positions_ref: ArtifactRef,
    rust_exports: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Validate all durable Phase 3 artifacts and return exact position rows."""

    return validate_phase3_artifacts(
        repository_root=repository_root,
        manifest=manifest,
        dataset_manifest_ref=dataset_manifest_ref,
        positions_ref=positions_ref,
        rust_exports=rust_exports,
    ).positions


def validate_phase3_artifacts(
    *,
    repository_root: Path,
    manifest: Mapping[str, Any],
    dataset_manifest_ref: ArtifactRef,
    positions_ref: ArtifactRef,
    rust_exports: Sequence[Mapping[str, Any]] | None = None,
) -> ValidatedPhase3Artifacts:
    """Validate and retain the exact canonical Phase 3 game and position rows."""

    root = repository_root.resolve(strict=True)
    try:
        expected_records = int(
            require_mapping(manifest.get("counts"), "Phase 3 counts")["positions"]
        )
        source_id = validate_production_dataset_manifest(
            dict(manifest),
            expected_records,
            repository_root=root,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError(f"invalid Phase 3 production manifest: {error}") from error
    parent = PurePosixPath(dataset_manifest_ref.path).parent
    if PurePosixPath(positions_ref.path).parent != parent:
        raise ContractError("Phase 3 positions must be a sibling of their dataset manifest")
    artifacts = require_mapping(manifest.get("artifacts"), "Phase 3 manifest.artifacts")
    games_ref = _manifest_artifact_ref(parent, artifacts, "games-00000.jsonl.gz")
    report_ref = _manifest_artifact_ref(parent, artifacts, "normalization-report.json")
    recorded_positions = _manifest_artifact_ref(parent, artifacts, "positions-00000.jsonl.gz")
    if recorded_positions != positions_ref:
        raise ContractError("Phase 3 positions disagree with their manifest artifact")

    # The report's derived metrics are intentionally not duplicated here. Its exact
    # bytes remain part of the hash chain while games/positions are independently checked.
    with verified_artifact_descriptor(root, report_ref, maximum_bytes=64 * 1024 * 1024):
        pass
    game_rows = _load_canonical_gzip_rows(
        root,
        games_ref,
        maximum_records=_MAX_GAMES,
        maximum_line_bytes=_MAX_GAME_LINE_BYTES,
    )
    position_rows = _load_canonical_gzip_rows(
        root,
        positions_ref,
        maximum_records=_MAX_POSITIONS,
        maximum_line_bytes=_MAX_POSITION_LINE_BYTES,
    )
    if len(game_rows) != _artifact_records(artifacts, "games-00000.jsonl.gz"):
        raise ContractError("Phase 3 games record count disagrees with its manifest")
    if len(position_rows) != _artifact_records(artifacts, "positions-00000.jsonl.gz"):
        raise ContractError("Phase 3 positions record count disagrees with its manifest")

    source = require_mapping(manifest.get("source"), "Phase 3 manifest.source")
    config = require_mapping(manifest.get("config"), "Phase 3 manifest.config")
    split_raw = require_mapping(config.get("split"), "Phase 3 manifest.config.split")
    split_policy = SplitPolicy(
        salt=require_string(split_raw, "salt", "Phase 3 split", maximum_length=1_024),
        validation_basis_points=require_int(
            split_raw, "validationBasisPoints", "Phase 3 split", minimum=0, maximum=10_000
        ),
        test_basis_points=require_int(
            split_raw, "testBasisPoints", "Phase 3 split", minimum=0, maximum=10_000
        ),
    )
    manifest_snapshots = require_list(
        manifest,
        "evidenceSnapshots",
        "Phase 3 manifest",
        minimum_items=1,
        maximum_items=10_000,
    )
    manifest_snapshot_sequence = [
        _snapshot_identity(snapshot, f"Phase 3 manifest evidence snapshot {index}")
        for index, snapshot in enumerate(manifest_snapshots)
    ]
    if manifest_snapshot_sequence != sorted(manifest_snapshot_sequence):
        raise ContractError("Phase 3 manifest evidence snapshots are not in canonical order")
    manifest_snapshot_identities = set(manifest_snapshot_sequence)
    if len(manifest_snapshot_identities) != len(manifest_snapshots):
        raise ContractError("Phase 3 manifest repeats an evidence snapshot tuple")
    required_evidence: dict[str, tuple[str, str]] = {}
    url_owners: dict[str, str] = {}
    path_owners: dict[str, str] = {}
    for index, snapshot in enumerate(manifest_snapshots):
        table = require_mapping(snapshot, f"Phase 3 manifest evidence snapshot {index}")
        evidence_id = require_string(
            table,
            "evidence_id",
            f"Phase 3 manifest evidence snapshot {index}",
            maximum_length=2_048,
        )
        url = require_string(
            table,
            "url",
            f"Phase 3 manifest evidence snapshot {index}",
            maximum_length=4_096,
        )
        content_type = require_string(
            table,
            "content_type",
            f"Phase 3 manifest evidence snapshot {index}",
            maximum_length=4_096,
        )
        path = require_string(
            table,
            "object_path",
            f"Phase 3 manifest evidence snapshot {index}",
            maximum_length=4_096,
        )
        previous = required_evidence.setdefault(evidence_id, (url, content_type))
        if previous != (url, content_type):
            raise ContractError("one Phase 3 evidence ID maps to multiple URL/content types")
        previous_url_owner = url_owners.setdefault(url, evidence_id)
        previous_path_owner = path_owners.setdefault(path, evidence_id)
        if previous_url_owner != evidence_id or previous_path_owner != evidence_id:
            raise ContractError("different Phase 3 evidence IDs reuse a URL/object path")

    catalog = _approved_catalog(root, source_id)
    game_by_id: dict[str, Mapping[str, Any]] = {}
    raw_to_game: dict[str, str] = {}
    observed_object_ids: set[str] = set()
    observed_snapshots: set[tuple[object, ...]] = set()
    for index, raw_game in enumerate(game_rows):
        context = f"Phase 3 game row {index + 1}"
        game = _validate_game(
            raw_game,
            context=context,
            source=source,
            source_id=source_id,
            catalog=catalog,
            split_policy=split_policy,
            terminal_tail_positions=require_int(
                config,
                "terminalTailPositions",
                "Phase 3 config",
                minimum=1,
                maximum=2_048,
            ),
            maximum_raw_bytes=require_int(
                config,
                "maxRawBytes",
                "Phase 3 config",
                minimum=1,
                maximum=16 * 1024 * 1024,
            ),
            manifest_snapshots=manifest_snapshot_identities,
            required_evidence=required_evidence,
        )
        game_id = str(game["canonicalSha256"])
        raw_object = require_mapping(game.get("rawObject"), f"{context}.rawObject")
        raw_sha256 = str(raw_object["sha256"])
        object_id = str(raw_object["objectId"])
        if game_id in game_by_id or raw_sha256 in raw_to_game or object_id in observed_object_ids:
            raise ContractError("Phase 3 games repeat a canonical/raw/catalog identity")
        game_by_id[game_id] = game
        raw_to_game[raw_sha256] = game_id
        observed_object_ids.add(object_id)
        for snapshot in require_list(
            require_mapping(game.get("licenseDecision"), f"{context}.licenseDecision"),
            "evidenceSnapshots",
            f"{context}.licenseDecision",
            minimum_items=1,
            maximum_items=10_000,
        ):
            observed_snapshots.add(_snapshot_identity(snapshot, f"{context} evidence snapshot"))

    canonical_manifest = set(_hash_array(manifest, "canonicalGameSha256"))
    raw_manifest = set(_hash_array(manifest, "rawObjectSha256"))
    if set(game_by_id) != canonical_manifest or set(raw_to_game) != raw_manifest:
        raise ContractError("Phase 3 games coverage differs from its manifest")
    if observed_snapshots != manifest_snapshot_identities:
        raise ContractError("Phase 3 game evidence union differs from its manifest")

    observed_positions: set[tuple[str, int]] = set()
    positions_by_game: dict[str, list[Mapping[str, Any]]] = {}
    parsed_positions: list[Mapping[str, Any]] = []
    for index, raw_position in enumerate(position_rows):
        context = f"Phase 3 position row {index + 1}"
        position = _validate_position(raw_position, context)
        key = (str(position["canonicalSha256"]), int(position["positionIndex"]))
        if key[0] not in game_by_id:
            raise ContractError(f"{context} is absent from its canonical games artifact")
        if key in observed_positions:
            raise ContractError("Phase 3 positions repeat a game/position identity")
        observed_positions.add(key)
        positions_by_game.setdefault(key[0], []).append(position)
        parsed_positions.append(position)
    expected_position_count = 0
    terminal_tail_positions = require_int(
        config,
        "terminalTailPositions",
        "Phase 3 config",
        minimum=1,
        maximum=2_048,
    )
    for game_id, game in game_by_id.items():
        rows = positions_by_game.get(game_id, [])
        _validate_game_positions(
            game,
            rows,
            terminal_tail_positions=terminal_tail_positions,
        )
        expected_position_count += int(game["positionCount"])
    if len(observed_positions) != expected_position_count:
        raise ContractError("Phase 3 positions do not exactly cover canonical game replay")

    if rust_exports is not None:
        _validate_rust_exports(game_rows, rust_exports)
        bind_positions_to_rust_exports(parsed_positions, game_rows, rust_exports)
    return ValidatedPhase3Artifacts(tuple(game_rows), tuple(parsed_positions))


def validate_phase3_rust_export_binding(
    artifacts: ValidatedPhase3Artifacts,
    exports: Sequence[Mapping[str, Any]],
) -> None:
    """Bind durable raw/canonical rows and successors to receipt-bound Rust output."""

    _validate_rust_exports(artifacts.games, exports)
    bind_positions_to_rust_exports(artifacts.positions, artifacts.games, exports)


def _manifest_artifact_ref(
    parent: PurePosixPath, artifacts: Mapping[str, Any], name: str
) -> ArtifactRef:
    table = require_mapping(artifacts.get(name), f"Phase 3 manifest artifact {name}")
    require_exact_keys(table, {"sha256", "size", "records"}, f"Phase 3 artifact {name}")
    return ArtifactRef(
        path=(parent / name).as_posix(),
        sha256=require_sha256(table, "sha256", f"Phase 3 artifact {name}"),
        size=require_int(
            table, "size", f"Phase 3 artifact {name}", minimum=1, maximum=_MAX_COMPRESSED_BYTES
        ),
    )


def _artifact_records(artifacts: Mapping[str, Any], name: str) -> int:
    return require_int(
        require_mapping(artifacts.get(name), f"Phase 3 manifest artifact {name}"),
        "records",
        f"Phase 3 manifest artifact {name}",
        minimum=1,
        maximum=_MAX_POSITIONS,
    )


def _load_canonical_gzip_rows(
    root: Path,
    reference: ArtifactRef,
    *,
    maximum_records: int,
    maximum_line_bytes: int,
) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    observed = 0
    try:
        with (
            verified_artifact_descriptor(
                root, reference, maximum_bytes=_MAX_COMPRESSED_BYTES
            ) as descriptor,
            os.fdopen(os.dup(descriptor), "rb") as raw_stream,
            gzip.GzipFile(fileobj=raw_stream, mode="rb") as stream,
        ):
            while line := stream.readline(maximum_line_bytes + 1):
                if len(rows) >= maximum_records:
                    raise ContractError("Phase 3 gzip JSONL exceeds its record bound")
                if len(line) > maximum_line_bytes or not line.endswith(b"\n"):
                    raise ContractError("Phase 3 gzip JSONL has an oversized/incomplete row")
                observed += len(line)
                if observed > _MAX_UNCOMPRESSED_BYTES:
                    raise ContractError("Phase 3 gzip JSONL exceeds its uncompressed bound")
                try:
                    value = json.loads(
                        line,
                        object_pairs_hook=_unique_json_object,
                        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
                    )
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                    raise ContractError("Phase 3 gzip JSONL contains invalid JSON") from error
                if not isinstance(value, dict):
                    raise ContractError("Phase 3 gzip JSONL row must be an object")
                if line[:-1] != canonical_json_bytes(value, newline=False):
                    raise ContractError("Phase 3 gzip JSONL row is not canonical JSON")
                rows.append(value)
    except (gzip.BadGzipFile, EOFError, OSError) as error:
        raise ContractError("cannot read bounded Phase 3 gzip JSONL") from error
    return rows


def _approved_catalog(root: Path, source_id: str) -> dict[str, tuple[str, str]]:
    registry_path = root / "configs/data_sources.yaml"
    if not registry_path.is_file():
        registry_path = Path(__file__).resolve().parents[3] / "configs/data_sources.yaml"
    try:
        source = load_source_registry(registry_path).get(source_id)
    except (OSError, RegistryError) as error:
        raise ContractError("Phase 3 source is absent from the approved registry") from error
    if not source.approved or not source.enabled or not source.machine_learning_allowed:
        raise ContractError("Phase 3 source is not approved for machine learning")
    return {item.object_id: (item.url, item.filename) for item in source.catalog}


def _validate_game(
    raw: Mapping[str, Any],
    *,
    context: str,
    source: Mapping[str, Any],
    source_id: str,
    catalog: Mapping[str, tuple[str, str]],
    split_policy: SplitPolicy,
    terminal_tail_positions: int,
    maximum_raw_bytes: int,
    manifest_snapshots: set[tuple[object, ...]],
    required_evidence: Mapping[str, tuple[str, str]],
) -> Mapping[str, Any]:
    game = require_mapping(raw, context)
    require_exact_keys(game, _GAME_KEYS, context)
    if game.get("schema") != "phase3_game/v1":
        raise ContractError(f"{context} has an unsupported schema")
    game_id = require_sha256(game, "gameId", context)
    if require_sha256(game, "canonicalSha256", context) != game_id:
        raise ContractError(f"{context} canonical identities disagree")
    raw_object = require_mapping(game.get("rawObject"), f"{context}.rawObject")
    require_exact_keys(raw_object, _RAW_KEYS, f"{context}.rawObject")
    raw_sha256 = require_sha256(raw_object, "sha256", f"{context}.rawObject")
    raw_csa = require_string(game, "rawCsa", context, maximum_length=maximum_raw_bytes)
    raw_bytes = raw_csa.encode("utf-8")
    if (
        hashlib.sha256(raw_bytes).hexdigest() != raw_sha256
        or require_int(
            raw_object,
            "size",
            f"{context}.rawObject",
            minimum=1,
            maximum=maximum_raw_bytes,
        )
        != len(raw_bytes)
        or raw_object.get("objectPath") != f"objects/sha256/{raw_sha256[:2]}/{raw_sha256}"
    ):
        raise ContractError(f"{context} raw CSA bytes/path disagree with their identity")
    object_id = require_string(raw_object, "objectId", f"{context}.rawObject", maximum_length=128)
    approved = catalog.get(object_id)
    original = require_string(
        raw_object, "originalFilename", f"{context}.rawObject", maximum_length=4_096
    )
    url = require_string(game, "url", context, maximum_length=4_096)
    if approved is None or approved != (url, original):
        raise ContractError(f"{context} raw object differs from the approved catalog")
    official_base = require_string(source, "officialBase", "Phase 3 source", maximum_length=4_096)
    if not _url_under_base(url, official_base):
        raise ContractError(f"{context} URL is outside its official source base")
    response = require_mapping(raw_object.get("response"), f"{context}.rawObject.response")
    require_exact_keys(response, _RAW_RESPONSE_KEYS, f"{context}.rawObject.response")
    for key in _RAW_RESPONSE_KEYS:
        require_optional_string(
            response, key, f"{context}.rawObject.response", maximum_length=4_096
        )

    normalized = require_string(game, "normalizedCsa", context, maximum_length=16 * 1024 * 1024)
    if hashlib.sha256(normalized.encode()).hexdigest() != game_id:
        raise ContractError(f"{context} normalized CSA differs from canonicalSha256")
    initial_sfen = require_string(game, "initialSfen", context, maximum_length=1_024)
    moves = require_list(game, "usiMoves", context, maximum_items=10_000)
    if any(not isinstance(move, str) or not move or len(move) > 16 for move in moves):
        raise ContractError(f"{context}.usiMoves is invalid")
    ply_count = require_int(game, "plyCount", context, minimum=0, maximum=10_000)
    position_count = require_int(game, "positionCount", context, minimum=1, maximum=10_001)
    if len(moves) != ply_count or position_count != ply_count + 1:
        raise ContractError(f"{context} move/position counts disagree")
    split = require_enum(game, "split", context, {"train", "validation", "test"})
    if split != assign_game_split(game_id, split_policy):
        raise ContractError(f"{context} split is not its salted hash assignment")
    if require_string(game, "sourceId", context) != source_id:
        raise ContractError(f"{context} sourceId differs from its manifest")
    outcome = require_enum(game, "outcome", context, {"black_win", "white_win", "draw", "unknown"})
    terminal = require_optional_string(game, "terminalReason", context, maximum_length=256)
    result_validation = require_enum(
        game, "resultValidation", context, {"verified", "external_condition", "missing"}
    )
    if (terminal is None) != (result_validation == "missing"):
        raise ContractError(f"{context} terminal/result-validation fields disagree")
    flags = require_mapping(game.get("flags"), f"{context}.flags")
    require_exact_keys(flags, _FLAGS_KEYS, f"{context}.flags")
    if require_bool(flags, "short", f"{context}.flags") != (ply_count < 20) or require_bool(
        flags, "long", f"{context}.flags"
    ) != (ply_count > 512):
        raise ContractError(f"{context} flags disagree with plyCount")
    players = require_mapping(game.get("players"), f"{context}.players")
    require_exact_keys(players, _PLAYERS_KEYS, f"{context}.players")
    for side in ("black", "white"):
        player = require_mapping(players.get(side), f"{context}.players.{side}")
        require_exact_keys(player, _PLAYER_KEYS, f"{context}.players.{side}")
        require_optional_string(player, "name", f"{context}.players.{side}", maximum_length=4_096)
        rating = player.get("rating")
        if rating is not None:
            require_int(player, "rating", f"{context}.players.{side}", minimum=0, maximum=1_000_000)
    for key in ("date", "sourceDateTime", "sourceTimeZone"):
        require_optional_string(game, key, context, maximum_length=4_096)
    validate_utc_timestamp(game.get("retrievedAt"), f"{context}.retrievedAt")

    decision = require_mapping(game.get("licenseDecision"), f"{context}.licenseDecision")
    require_exact_keys(decision, _LICENSE_DECISION_KEYS, f"{context}.licenseDecision")
    if (
        decision.get("license") != source.get("license")
        or decision.get("evidence") != source.get("licenseEvidence")
        or decision.get("redistributable") != source.get("redistributable")
        or decision.get("machineLearningAllowed") != source.get("machineLearningAllowed")
    ):
        raise ContractError(f"{context} license decision differs from its manifest")
    snapshots = require_list(
        decision,
        "evidenceSnapshots",
        f"{context}.licenseDecision",
        minimum_items=len(required_evidence),
        maximum_items=len(required_evidence),
    )
    per_id: dict[str, tuple[object, ...]] = {}
    snapshot_urls: set[str] = set()
    for snapshot_index, snapshot in enumerate(snapshots):
        identity = _snapshot_identity(snapshot, f"{context} evidence {snapshot_index}")
        table = require_mapping(snapshot, f"{context} evidence {snapshot_index}")
        evidence_id = str(table["evidence_id"])
        url_value = str(table["url"])
        content_type = str(table["content_type"])
        if (
            identity not in manifest_snapshots
            or required_evidence.get(evidence_id) != (url_value, content_type)
            or evidence_id in per_id
        ):
            raise ContractError(f"{context} evidence is missing, duplicated, or unapproved")
        per_id[evidence_id] = identity
        snapshot_urls.add(url_value)
    if set(per_id) != set(required_evidence):
        raise ContractError(f"{context} does not cover every required evidence ID")
    license_evidence = require_list(
        source, "licenseEvidence", "Phase 3 source", minimum_items=1, maximum_items=100
    )
    for item in license_evidence:
        evidence = require_mapping(item, "Phase 3 source license evidence")
        if evidence.get("url") not in snapshot_urls:
            raise ContractError(f"{context} lacks a snapshot for approved license evidence")

    del initial_sfen, position_count, outcome, terminal, terminal_tail_positions
    return game


def _validate_game_positions(
    game: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    terminal_tail_positions: int,
) -> None:
    game_id = str(game["canonicalSha256"])
    ordered = sorted(rows, key=lambda row: int(row["positionIndex"]))
    ply_count = int(game["plyCount"])
    if len(ordered) != ply_count + 1 or [row["positionIndex"] for row in ordered] != list(
        range(ply_count + 1)
    ):
        raise ContractError(f"Phase 3 game {game_id} positions are not contiguous from zero")
    if ordered[0]["sfen"] != game["initialSfen"]:
        raise ContractError(f"Phase 3 game {game_id} initial SFEN differs from position zero")
    moves = list(game["usiMoves"])
    raw_sha256 = require_mapping(game.get("rawObject"), "Phase 3 game.rawObject")["sha256"]
    tail_start = max(0, ply_count - terminal_tail_positions)
    for index, row in enumerate(ordered):
        expected_move = moves[index] if index < ply_count else None
        expected_next = ordered[index + 1]["sfen"] if index < ply_count else None
        terminal_tail = index >= tail_start
        if (
            row["gameId"] != game_id
            or row["canonicalSha256"] != game_id
            or row["rawSha256"] != raw_sha256
            or row["sourceId"] != game["sourceId"]
            or row["split"] != game["split"]
            or row["moveUsi"] != expected_move
            or row["nextSfen"] != expected_next
            or row["outcome"] != game["outcome"]
            or row["terminalReason"] != game["terminalReason"]
            or row["fullPlies"] != ply_count
            or row["remainingPlies"] != ply_count - index
            or row["terminalTail"] != terminal_tail
            or row["eligible"] != (expected_move is not None and not terminal_tail)
        ):
            raise ContractError(
                f"Phase 3 game {game_id} position {index} differs from its game sequence"
            )


def _validate_position(raw: Mapping[str, Any], context: str) -> Mapping[str, Any]:
    row = require_mapping(raw, context)
    require_exact_keys(row, _POSITION_KEYS, context)
    if row.get("schema") != "phase3_position/v1":
        raise ContractError(f"{context} has an unsupported schema")
    game_id = require_sha256(row, "gameId", context)
    if require_sha256(row, "canonicalSha256", context) != game_id:
        raise ContractError(f"{context} game identities disagree")
    require_sha256(row, "rawSha256", context)
    require_string(row, "sourceId", context)
    require_enum(row, "split", context, {"train", "validation", "test"})
    index = require_int(row, "positionIndex", context, minimum=0, maximum=10_000)
    sfen = require_string(row, "sfen", context, maximum_length=1_024)
    fields = sfen.split(" ")
    if len(fields) != 4 or fields[1] not in {"b", "w"}:
        raise ContractError(f"{context}.sfen is invalid")
    move = require_optional_string(row, "moveUsi", context, maximum_length=16)
    next_sfen = require_optional_string(row, "nextSfen", context, maximum_length=1_024)
    if (move is None) != (next_sfen is None):
        raise ContractError(f"{context} move/next-SFEN presence differs")
    require_enum(row, "outcome", context, {"black_win", "white_win", "draw", "unknown"})
    require_optional_string(row, "terminalReason", context, maximum_length=256)
    side = require_enum(row, "sideToMove", context, {"black", "white"})
    if side != ("black" if fields[1] == "b" else "white"):
        raise ContractError(f"{context}.sideToMove disagrees with SFEN")
    full = require_int(row, "fullPlies", context, minimum=0, maximum=10_000)
    remaining = require_int(row, "remainingPlies", context, minimum=0, maximum=10_000)
    if full - index != remaining:
        raise ContractError(f"{context} ply counts disagree")
    eligible = require_bool(row, "eligible", context)
    tail = require_bool(row, "terminalTail", context)
    if eligible and (tail or move is None):
        raise ContractError(f"{context} eligibility is inconsistent")
    if move is None and (index != full or remaining != 0 or eligible or not tail):
        raise ContractError(f"{context} final position fields are inconsistent")
    return row


def _validate_rust_exports(
    games: Sequence[Mapping[str, Any]], exports: Sequence[Mapping[str, Any]]
) -> None:
    if len(exports) != len(games):
        raise ContractError("receipt-bound Rust export count differs from Phase 3 games")
    for index, (game, raw_export) in enumerate(zip(games, exports, strict=True)):
        context = f"receipt-bound Rust export row {index + 1}"
        export = require_mapping(raw_export, context)
        require_exact_keys(export, _EXPORT_KEYS, context)
        if export.get("schema") != "phase3_csa_export/v1" or export.get("status") != "ok":
            raise ContractError(f"{context} rejected a Phase 3 raw CSA record")
        if export.get("inputFile") != f"game-{index + 1:06d}.csa":
            raise ContractError(f"{context} belongs to a different staged raw CSA record")
        players = require_mapping(game.get("players"), f"Phase 3 game {index + 1}.players")
        expected = {
            "normalizedCsa": game["normalizedCsa"],
            "initialSfen": game["initialSfen"],
            "usiMoves": game["usiMoves"],
            "blackName": require_mapping(players.get("black"), "black player").get("name"),
            "whiteName": require_mapping(players.get("white"), "white player").get("name"),
            "terminalReason": game["terminalReason"],
            "resultValidation": game["resultValidation"],
        }
        if any(export.get(key) != value for key, value in expected.items()) or not (
            export.get("outcome") == game["outcome"]
            or (
                # The approved Phase 3 exporter conservatively retained two verified
                # ordinary repetitions as unknown; current Rust replay proves them draws.
                game["outcome"] == "unknown"
                and export.get("outcome") == "draw"
                and game["terminalReason"] == "SENNICHITE"
                and game["resultValidation"] == "verified"
            )
        ):
            raise ContractError(f"{context} differs from the durable Phase 3 game")
        sfens = require_list(
            export,
            "positionSfens",
            context,
            minimum_items=int(game["positionCount"]),
            maximum_items=int(game["positionCount"]),
        )
        if any(not isinstance(sfen, str) for sfen in sfens):
            raise ContractError(f"{context}.positionSfens is invalid")


def bind_positions_to_rust_exports(
    positions: Sequence[Mapping[str, Any]],
    games: Sequence[Mapping[str, Any]],
    exports: Sequence[Mapping[str, Any]],
) -> None:
    """Bind every Python position successor to receipt-bound Rust replay output."""

    _validate_rust_exports(games, exports)
    expected: dict[tuple[str, int], tuple[str, str | None, str | None]] = {}
    for game, export in zip(games, exports, strict=True):
        game_id = str(game["canonicalSha256"])
        sfens = list(export["positionSfens"])
        moves = list(export["usiMoves"])
        for index, sfen in enumerate(sfens):
            expected[(game_id, index)] = (
                str(sfen),
                str(moves[index]) if index < len(moves) else None,
                str(sfens[index + 1]) if index + 1 < len(sfens) else None,
            )
    for row in positions:
        key = (str(row["canonicalSha256"]), int(row["positionIndex"]))
        if expected.get(key) != (row["sfen"], row["moveUsi"], row["nextSfen"]):
            raise ContractError("Phase 3 move/next-SFEN differs from receipt-bound Rust replay")


def _snapshot_identity(value: object, context: str) -> tuple[object, ...]:
    snapshot = require_mapping(value, context)
    require_exact_keys(
        snapshot,
        {"evidence_id", "url", "retrieved_at", "sha256", "size", "content_type", "object_path"},
        context,
    )
    evidence_id = require_string(snapshot, "evidence_id", context, maximum_length=2_048)
    url = require_string(snapshot, "url", context, maximum_length=4_096)
    retrieved = require_string(snapshot, "retrieved_at", context, maximum_length=64)
    validate_utc_timestamp(retrieved, f"{context}.retrieved_at")
    sha256 = require_sha256(snapshot, "sha256", context)
    size = require_int(snapshot, "size", context, minimum=1, maximum=16 * 1024 * 1024)
    content_type = require_string(snapshot, "content_type", context, maximum_length=4_096)
    path = require_string(snapshot, "object_path", context, maximum_length=4_096)
    if path != f"evidence/sha256/{sha256[:2]}/{sha256}":
        raise ContractError(f"{context} path is not content-addressed")
    if not url.startswith(("https://", "http://")):
        raise ContractError(f"{context}.url is invalid")
    return evidence_id, url, retrieved, sha256, size, content_type, path


def _hash_array(manifest: Mapping[str, Any], key: str) -> list[str]:
    values = require_list(manifest, key, "Phase 3 manifest", minimum_items=1, maximum_items=100)
    result = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or len(value) != 64:
            raise ContractError(f"Phase 3 manifest.{key}[{index}] is invalid")
        result.append(value)
    return result


def _url_under_base(url: str, base: str) -> bool:
    url_parts = urlsplit(url)
    base_parts = urlsplit(base)
    base_path = base_parts.path.rstrip("/")
    return (
        url_parts.scheme == base_parts.scheme
        and url_parts.netloc == base_parts.netloc
        and (url_parts.path == base_path or url_parts.path.startswith(f"{base_path}/"))
        and not url_parts.query
        and not url_parts.fragment
    )


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result
