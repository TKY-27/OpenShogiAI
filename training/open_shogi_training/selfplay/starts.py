"""Deterministic Phase 3 start-set selection and Rust legality evidence."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final

from open_shogi_training.data.aobazero import (
    AobaZeroAdaptationError,
    adapt_aobazero_csa,
)
from open_shogi_training.labeling.artifacts import iter_jsonl_descriptor_records
from open_shogi_training.models.dataset import validate_production_dataset_manifest

from .common import (
    ArtifactRef,
    ContractError,
    artifact_ref,
    canonical_sha256,
    contained_path,
    ensure_contained_directory,
    load_json_artifact,
    require_bool,
    require_clean_head,
    require_enum,
    require_exact_keys,
    require_identifier,
    require_int,
    require_mapping,
    require_optional_string,
    require_sha256,
    require_string,
    validate_relative_path,
    verified_artifact_descriptor,
    verify_artifact_ref,
    write_bytes_new,
)
from .config import MAX_JSON_SAFE_INTEGER, PHASE6_SEED
from .engine_receipt import validate_engine_build_receipt
from .execution import CommandRunner
from .phase3 import (
    validate_phase3_artifact_chain,
    validate_phase3_artifacts,
    validate_phase3_rust_export_binding,
)
from .planning import (
    MAX_START_POSITIONS,
    START_POSITIONS_SCHEMA,
    START_VALIDATION_SCHEMA,
    StartPositionSet,
    parse_start_positions,
    start_position_identity,
)

MAX_PHASE3_POSITIONS: Final = 250_000
MAX_SELECTED_PER_SPLIT: Final = 5_000
_MAX_PHASE3_EXPORT_BYTES: Final = 512 * 1024 * 1024
_MAX_PHASE3_EXPORT_LINE_BYTES: Final = 32 * 1024 * 1024
_MAX_PHASE3_EXPORT_ATTEMPTS: Final = 10

_PHASE3_POSITION_KEYS = frozenset(
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
_DATASET_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "datasetId",
        "source",
        "config",
        "counts",
        "artifacts",
        "rawObjectSha256",
        "canonicalGameSha256",
        "evidenceSnapshots",
    }
)


def build_start_positions_from_phase3(
    *,
    repository_root: Path,
    positions_ref: ArtifactRef,
    dataset_manifest_ref: ArtifactRef,
    seed: int,
    train_count: int,
    validation_count: int,
) -> dict[str, object]:
    """Select eligible train/validation positions without cross-split state leakage."""

    if (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or not 0 <= seed <= MAX_JSON_SAFE_INTEGER
    ):
        raise ContractError("start-position seed is outside the JSON-safe integer range")
    if seed != PHASE6_SEED:
        raise ContractError(f"Phase 6 start-position seed must be {PHASE6_SEED}")
    for name, count in (("train_count", train_count), ("validation_count", validation_count)):
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or not 1 <= count <= MAX_SELECTED_PER_SPLIT
        ):
            raise ContractError(f"{name} must be in 1..{MAX_SELECTED_PER_SPLIT}")
    if train_count + validation_count > MAX_START_POSITIONS:
        raise ContractError("selected start-position count exceeds its global limit")

    manifest = load_json_artifact(repository_root, dataset_manifest_ref)
    expected_records = validate_phase3_dataset_binding(
        manifest,
        positions_ref=positions_ref,
        dataset_manifest_ref=dataset_manifest_ref,
        repository_root=repository_root,
    )
    validated_rows = validate_phase3_artifact_chain(
        repository_root=repository_root,
        manifest=require_mapping(manifest, "Phase 3 dataset manifest"),
        dataset_manifest_ref=dataset_manifest_ref,
        positions_ref=positions_ref,
    )
    if len(validated_rows) != expected_records:
        raise ContractError("Phase 3 full-chain position count disagrees with its manifest")
    manifest_root = require_mapping(manifest, "Phase 3 dataset manifest")
    manifest_source = require_mapping(
        manifest_root.get("source"), "Phase 3 dataset manifest.source"
    )
    source_id = require_identifier(manifest_source, "sourceId", "Phase 3 dataset manifest.source")
    canonical_games = set(manifest_root["canonicalGameSha256"])
    raw_objects = set(manifest_root["rawObjectSha256"])

    candidates: dict[str, dict[str, tuple[str, int]]] = {}
    observed_game_splits: dict[str, str] = {}
    observed_game_raw: dict[str, str] = {}
    observed_positions: set[tuple[str, int]] = set()
    observed_rows = 0
    for raw_row in validated_rows:
        observed_rows += 1
        parsed = _parse_phase3_position(raw_row, observed_rows)
        game_id = str(parsed["canonicalSha256"])
        raw_sha256 = str(parsed["rawSha256"])
        split = str(parsed["split"])
        position_key = (game_id, int(parsed["positionIndex"]))
        if parsed["sourceId"] != source_id:
            raise ContractError("Phase 3 position source differs from approved manifest")
        if game_id not in canonical_games or raw_sha256 not in raw_objects:
            raise ContractError("Phase 3 position game/raw hash is absent from its manifest")
        if position_key in observed_positions:
            raise ContractError("Phase 3 positions repeat a game/position index")
        observed_positions.add(position_key)
        previous_split = observed_game_splits.setdefault(game_id, split)
        previous_raw = observed_game_raw.setdefault(game_id, raw_sha256)
        if previous_split != split:
            raise ContractError("one Phase 3 game appears in multiple data splits")
        if previous_raw != raw_sha256:
            raise ContractError("one Phase 3 game refers to multiple raw objects")
        if not parsed["eligible"]:
            continue
        canonical_sfen = _reset_move_number(str(parsed["sfen"]))
        origin = (game_id, int(parsed["positionIndex"]))
        by_split = candidates.setdefault(canonical_sfen, {})
        previous = by_split.get(split)
        if previous is None or origin < previous:
            by_split[split] = origin
    if observed_rows != expected_records:
        raise ContractError("Phase 3 position count disagrees with the dataset manifest")
    if (
        set(observed_game_splits) != canonical_games
        or set(observed_game_raw.values()) != raw_objects
    ):
        raise ContractError("Phase 3 position game/raw coverage differs from its manifest")

    cross_split = {sfen for sfen, origins in candidates.items() if len(origins) > 1}
    split_candidates: dict[str, list[tuple[str, str, int]]] = {
        "train": [],
        "validation": [],
    }
    for sfen, origins in candidates.items():
        if sfen in cross_split:
            continue
        split, (game_id, position_index) = next(iter(origins.items()))
        if split == "test":
            continue
        split_candidates[split].append((sfen, game_id, position_index))

    selected: list[dict[str, object]] = []
    for split, count in (("train", train_count), ("validation", validation_count)):
        ranked = sorted(
            split_candidates[split],
            key=lambda item: (
                hashlib.sha256(
                    f"phase6-start\0{seed}\0{split}\0{item[0]}\0{item[1]}\0{item[2]}".encode()
                ).digest(),
                item,
            ),
        )
        if len(ranked) < count:
            raise ContractError(
                f"Phase 3 data has only {len(ranked)} safe {split} start positions; "
                f"{count} required"
            )
        for sfen, game_id, position_index in ranked[:count]:
            identity = start_position_identity(game_id, position_index, sfen)
            selected.append(
                {
                    "positionId": identity,
                    "sfen": sfen,
                    "sourceGameSha256": game_id,
                    "positionIndex": position_index,
                    "split": split,
                }
            )
    selected.sort(key=lambda row: str(row["positionId"]))
    result: dict[str, object] = {
        "schema": START_POSITIONS_SCHEMA,
        "datasetManifest": dataset_manifest_ref.as_dict(),
        "sourcePositions": positions_ref.as_dict(),
        "selection": {
            "seed": seed,
            "trainCount": train_count,
            "validationCount": validation_count,
            "requireEligible": True,
            "resetMoveNumber": True,
            "excludedCrossSplitStates": len(cross_split),
        },
        "positions": selected,
    }
    parse_start_positions(result)
    return result


def revalidate_phase3_source_with_engine(
    *,
    repository_root: Path,
    start_positions: StartPositionSet,
    engine_ref: ArtifactRef,
    engine_build_receipt: ArtifactRef,
    git_commit: str,
) -> None:
    """Reprove raw CSA canonicalization and every successor with the pinned Rust engine."""

    root = repository_root.resolve(strict=True)
    require_clean_head(root, git_commit)
    validate_engine_build_receipt(
        root,
        engine_build_receipt,
        expected_engine=engine_ref,
        expected_git_commit=git_commit,
    )
    manifest = require_mapping(
        load_json_artifact(root, start_positions.dataset_manifest),
        "Phase 3 dataset manifest",
    )
    artifacts = validate_phase3_artifacts(
        repository_root=root,
        manifest=manifest,
        dataset_manifest_ref=start_positions.dataset_manifest,
        positions_ref=start_positions.source_positions,
    )
    if not artifacts.games:
        raise ContractError("Phase 3 Rust revalidation requires at least one game")
    identity = canonical_sha256(
        {
            "schema": "phase3_receipt_bound_rust_revalidation/v2",
            "adapter": "aobazero_csa/v1",
            "datasetManifest": start_positions.dataset_manifest.as_dict(),
            "sourcePositions": start_positions.source_positions.as_dict(),
            "engine": engine_ref.as_dict(),
            "engineBuildReceipt": engine_build_receipt.as_dict(),
            "gitCommit": git_commit,
        }
    )
    base = f"local/phase3-revalidation/{identity}"
    input_dir = f"{base}/input"
    ensure_contained_directory(root, input_dir)
    for index, game in enumerate(artifacts.games, start=1):
        adapted_csa = _adapt_phase3_game_csa(game, index=index)
        relative = f"{input_dir}/game-{index:06d}.csa"
        expected = ArtifactRef(
            relative,
            hashlib.sha256(adapted_csa).hexdigest(),
            len(adapted_csa),
        )
        destination = contained_path(root, relative)
        if destination.exists():
            if artifact_ref(root, relative, maximum_bytes=16 * 1024 * 1024) != expected:
                raise ContractError("staged adapted Phase 3 CSA differs from its immutable input")
        else:
            observed_sha256, observed_size = write_bytes_new(destination, adapted_csa)
            if (observed_sha256, observed_size) != (expected.sha256, expected.size):
                raise ContractError(
                    "staged adapted Phase 3 CSA identity changed during publication"
                )

    attempt, output_path, stdout_path, stderr_path, receipt_path = _phase3_export_attempt(
        root, base
    )
    del attempt
    ensure_contained_directory(root, str(PurePosixPath(output_path).parent))
    outcome = CommandRunner(root, require_clean_repository=True).run(
        {
            "kind": "engine_dataset_export",
            "argv": [
                engine_ref.path,
                "export-csa-jsonl",
                "--input-dir",
                input_dir,
                "--output",
                output_path,
                "--max-games",
                str(len(artifacts.games)),
            ],
            "timeoutSeconds": require_int(
                require_mapping(manifest.get("config"), "Phase 3 config"),
                "exporterTimeoutSeconds",
                "Phase 3 config",
                minimum=1,
                maximum=3_600,
            ),
        },
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        expected_executable=engine_ref,
        engine_build_receipt=engine_build_receipt,
        memory_limit_mib=1_024,
        receipt_path=receipt_path,
    )
    if (
        outcome.return_code != 0
        or outcome.timed_out
        or outcome.output_limit_exceeded
        or outcome.memory_limit_exceeded
    ):
        raise ContractError("receipt-bound Rust Phase 3 exporter failed")
    exports = _load_phase3_exports(root, output_path, expected=len(artifacts.games))
    validate_phase3_rust_export_binding(artifacts, exports)


def _adapt_phase3_game_csa(game: Mapping[str, Any], *, index: int) -> bytes:
    """Re-derive one immutable Rust input from its exact acquired AobaZero bytes."""

    context = f"Phase 3 game {index}"
    raw_object = require_mapping(game.get("rawObject"), f"{context}.rawObject")
    expected_sha256 = require_sha256(raw_object, "sha256", f"{context}.rawObject")
    expected_size = require_int(
        raw_object,
        "size",
        f"{context}.rawObject",
        minimum=1,
        maximum=16 * 1024 * 1024,
    )
    raw_bytes = require_string(
        game,
        "rawCsa",
        context,
        maximum_length=16 * 1024 * 1024,
    ).encode("utf-8")
    if len(raw_bytes) != expected_size or hashlib.sha256(raw_bytes).hexdigest() != expected_sha256:
        raise ContractError(f"{context} raw CSA bytes differ from their durable identity")
    try:
        adapted = adapt_aobazero_csa(raw_bytes)
    except AobaZeroAdaptationError as error:
        raise ContractError(f"{context} AobaZero adaptation failed: {error}") from error
    if adapted.raw_sha256 != expected_sha256:
        raise ContractError(f"{context} AobaZero adapter raw identity mismatch")
    return adapted.csa.encode("utf-8")


def _phase3_export_attempt(root: Path, base: str) -> tuple[int, str, str, str, str]:
    for attempt in range(1, _MAX_PHASE3_EXPORT_ATTEMPTS + 1):
        attempt_root = f"{base}/attempt-{attempt:03d}"
        output = f"{attempt_root}/exports.jsonl"
        stdout = f"{attempt_root}/stdout.log"
        stderr = f"{attempt_root}/stderr.log"
        receipt = f"{attempt_root}/command-receipt.json"
        paths = tuple(contained_path(root, path) for path in (output, stdout, stderr, receipt))
        if paths[-1].exists():
            return attempt, output, stdout, stderr, receipt
        if not any(path.exists() or path.is_symlink() for path in paths):
            return attempt, output, stdout, stderr, receipt
    raise ContractError("Phase 3 Rust revalidation exhausted its bounded recovery attempts")


def _load_phase3_exports(
    root: Path,
    relative: str,
    *,
    expected: int,
) -> tuple[Mapping[str, Any], ...]:
    reference = artifact_ref(root, relative, maximum_bytes=_MAX_PHASE3_EXPORT_BYTES)
    try:
        with verified_artifact_descriptor(
            root,
            reference,
            maximum_bytes=_MAX_PHASE3_EXPORT_BYTES,
        ) as descriptor:
            rows = tuple(
                iter_jsonl_descriptor_records(
                    descriptor,
                    display_path=Path(relative),
                    max_bytes=_MAX_PHASE3_EXPORT_BYTES,
                    max_line_bytes=_MAX_PHASE3_EXPORT_LINE_BYTES,
                    max_records=expected,
                )
            )
    except (OSError, ValueError) as error:
        raise ContractError(f"cannot read receipt-bound Phase 3 Rust exports: {error}") from error
    if len(rows) != expected:
        raise ContractError("receipt-bound Rust Phase 3 export count is incomplete")
    return tuple(
        require_mapping(row, f"Phase 3 Rust export {index}") for index, row in enumerate(rows)
    )


def execute_start_position_validation(
    *,
    repository_root: Path,
    start_positions: object,
    start_positions_ref: ArtifactRef,
    engine_ref: ArtifactRef,
    git_commit: str,
    output_root: str,
    timeout_seconds: int,
    runner: CommandRunner,
    now: Callable[[], str] | None = None,
    engine_build_receipt: ArtifactRef | None = None,
) -> dict[str, object]:
    """Run bounded Rust perft-depth-0 checks and bind immutable logs as evidence."""

    starts = validate_start_position_source_artifact(
        repository_root=repository_root,
        start_positions=start_positions,
        start_positions_ref=start_positions_ref,
    )
    verify_artifact_ref(repository_root, engine_ref)
    if PurePosixPath(engine_ref.path).name != "open-shogi-cli":
        raise ContractError("start validation requires the repository open-shogi-cli")
    output_root = validate_relative_path(output_root)
    if not 1 <= timeout_seconds <= 3_600:
        raise ContractError("start validation timeout must be in 1..3600 seconds")
    if (
        not isinstance(git_commit, str)
        or not 7 <= len(git_commit) <= 64
        or any(character not in "0123456789abcdef" for character in git_commit)
    ):
        raise ContractError("git commit must be a 7..64 character lowercase hexadecimal object ID")
    if isinstance(runner, CommandRunner) and runner.require_clean_repository:
        require_clean_head(repository_root, git_commit)
    if isinstance(runner, CommandRunner) and runner.require_engine_build_receipt:
        if engine_build_receipt is None:
            raise ContractError("production start validation requires an engine build receipt")
        validate_engine_build_receipt(
            repository_root,
            engine_build_receipt,
            expected_engine=engine_ref,
            expected_git_commit=git_commit,
        )
    timestamp = now or _utc_now
    results: list[dict[str, object]] = []
    for position in starts.positions:
        command = {
            "kind": "engine_validation",
            "argv": [
                engine_ref.path,
                "perft",
                "--depth",
                "0",
                "--sfen",
                position.sfen,
            ],
            "timeoutSeconds": timeout_seconds,
        }
        log_base = _available_log_base(
            repository_root,
            f"{output_root}/logs/{position.position_id}",
        )
        outcome = runner.run(
            command,
            stdout_path=f"{log_base}.stdout.log",
            stderr_path=f"{log_base}.stderr.log",
            expected_executable=engine_ref,
            engine_build_receipt=engine_build_receipt,
            memory_limit_mib=1_024,
        )
        legal = (
            outcome.return_code == 0
            and not outcome.timed_out
            and not outcome.output_limit_exceeded
            and not outcome.memory_limit_exceeded
        )
        results.append(
            {
                "positionId": position.position_id,
                "sfenSha256": hashlib.sha256(position.sfen.encode("utf-8")).hexdigest(),
                "legal": legal,
                "returnCode": outcome.return_code,
                "timedOut": outcome.timed_out,
                "outputLimitExceeded": outcome.output_limit_exceeded,
                "memoryLimitExceeded": outcome.memory_limit_exceeded,
                "peakRssBytes": outcome.peak_rss_bytes,
                "rssMeasurement": outcome.rss_measurement,
                "stdout": outcome.stdout.as_dict(),
                "stderr": outcome.stderr.as_dict(),
                "completedAt": timestamp(),
            }
        )
    return {
        "schema": START_VALIDATION_SCHEMA,
        "startPositions": start_positions_ref.as_dict(),
        "engine": engine_ref.as_dict(),
        "engineBuildReceipt": (
            engine_build_receipt.as_dict() if engine_build_receipt is not None else None
        ),
        "gitCommit": git_commit,
        "method": "open-shogi-cli-perft-depth-0",
        "results": results,
    }


def validate_start_position_source_artifact(
    *,
    repository_root: Path,
    start_positions: object,
    start_positions_ref: ArtifactRef,
) -> StartPositionSet:
    """Rebuild a start set from its hashed Phase 3 source and require exact equality."""

    starts = parse_start_positions(start_positions)
    if load_json_artifact(repository_root, start_positions_ref) != start_positions:
        raise ContractError("start-position value differs from its referenced artifact")
    verify_artifact_ref(repository_root, starts.dataset_manifest)
    verify_artifact_ref(repository_root, starts.source_positions)
    root = require_mapping(start_positions, "start positions")
    selection = require_mapping(root.get("selection"), "start positions.selection")
    rebuilt = build_start_positions_from_phase3(
        repository_root=repository_root,
        positions_ref=starts.source_positions,
        dataset_manifest_ref=starts.dataset_manifest,
        seed=require_int(
            selection,
            "seed",
            "start positions.selection",
            minimum=0,
            maximum=MAX_JSON_SAFE_INTEGER,
        ),
        train_count=require_int(
            selection, "trainCount", "start positions.selection", minimum=1, maximum=5_000
        ),
        validation_count=require_int(
            selection,
            "validationCount",
            "start positions.selection",
            minimum=1,
            maximum=5_000,
        ),
    )
    if rebuilt != start_positions:
        raise ContractError("start-position artifact is not its deterministic Phase 3 selection")
    return starts


def validate_phase3_dataset_binding(
    raw: object,
    *,
    positions_ref: ArtifactRef,
    dataset_manifest_ref: ArtifactRef,
    repository_root: Path | None = None,
) -> int:
    root = require_mapping(raw, "Phase 3 dataset manifest")
    require_exact_keys(root, _DATASET_MANIFEST_KEYS, "Phase 3 dataset manifest")
    if root.get("schema") != "phase3_dataset_manifest/v1":
        raise ContractError("unsupported Phase 3 dataset manifest schema")
    manifest_parent = PurePosixPath(dataset_manifest_ref.path).parent
    positions_path = PurePosixPath(positions_ref.path)
    if positions_path.parent != manifest_parent:
        raise ContractError("Phase 3 positions must be a sibling of their dataset manifest")
    artifacts = require_mapping(root.get("artifacts"), "Phase 3 dataset manifest.artifacts")
    artifact = require_mapping(
        artifacts.get(positions_path.name),
        f"Phase 3 dataset manifest.artifacts.{positions_path.name}",
    )
    require_exact_keys(
        artifact,
        {"sha256", "size", "records"},
        f"Phase 3 dataset manifest.artifacts.{positions_path.name}",
    )
    if require_sha256(artifact, "sha256", "Phase 3 position artifact") != positions_ref.sha256:
        raise ContractError("Phase 3 positions SHA-256 disagrees with the dataset manifest")
    if (
        require_int(
            artifact,
            "size",
            "Phase 3 position artifact",
            minimum=0,
            maximum=4 * 1024 * 1024 * 1024,
        )
        != positions_ref.size
    ):
        raise ContractError("Phase 3 positions size disagrees with the dataset manifest")
    records = require_int(
        artifact,
        "records",
        "Phase 3 position artifact",
        minimum=1,
        maximum=MAX_PHASE3_POSITIONS,
    )
    try:
        validate_production_dataset_manifest(dict(root), records, repository_root=repository_root)
    except ValueError as error:
        raise ContractError(f"invalid Phase 3 production manifest: {error}") from error
    return records


def _parse_phase3_position(raw: object, row_number: int) -> Mapping[str, Any]:
    context = f"Phase 3 position row {row_number}"
    row = require_mapping(raw, context)
    require_exact_keys(row, _PHASE3_POSITION_KEYS, context)
    if row.get("schema") != "phase3_position/v1":
        raise ContractError(f"{context} has an unsupported schema")
    game_id = require_sha256(row, "gameId", context)
    if require_sha256(row, "canonicalSha256", context) != game_id:
        raise ContractError(f"{context} game identities disagree")
    require_sha256(row, "rawSha256", context)
    require_identifier(row, "sourceId", context)
    require_enum(row, "split", context, {"train", "validation", "test"})
    index = require_int(row, "positionIndex", context, minimum=0, maximum=10_000)
    sfen = require_string(row, "sfen", context, maximum_length=1_024)
    fields = sfen.split(" ")
    if (
        len(fields) != 4
        or any(not field for field in fields)
        or fields[1] not in {"b", "w"}
        or not fields[3].isascii()
        or not fields[3].isdecimal()
        or int(fields[3]) < 1
    ):
        raise ContractError(f"{context}.sfen is invalid")
    require_optional_string(row, "moveUsi", context, maximum_length=16)
    require_optional_string(row, "nextSfen", context, maximum_length=1_024)
    require_enum(row, "outcome", context, {"black_win", "white_win", "draw", "unknown"})
    require_optional_string(row, "terminalReason", context, maximum_length=256)
    side = require_enum(row, "sideToMove", context, {"black", "white"})
    expected_side = "black" if fields[1] == "b" else "white"
    if side != expected_side:
        raise ContractError(f"{context}.sideToMove disagrees with SFEN")
    full_plies = require_int(row, "fullPlies", context, minimum=0, maximum=10_000)
    remaining = require_int(row, "remainingPlies", context, minimum=0, maximum=10_000)
    if full_plies - index != remaining:
        raise ContractError(f"{context} ply counts disagree")
    eligible = require_bool(row, "eligible", context)
    terminal_tail = require_bool(row, "terminalTail", context)
    if eligible and (terminal_tail or row.get("moveUsi") is None):
        raise ContractError(f"{context} has inconsistent eligibility")
    return row


def _reset_move_number(sfen: str) -> str:
    fields = sfen.split(" ")
    return " ".join((*fields[:3], "1"))


def _available_log_base(repository_root: Path, base: str) -> str:
    """Choose fresh immutable log paths after an interrupted prior invocation."""

    candidates = [base, *(f"{base}.recovery-{index:03d}" for index in range(1, 101))]
    for candidate in candidates:
        stdout = contained_path(repository_root, f"{candidate}.stdout.log")
        stderr = contained_path(repository_root, f"{candidate}.stderr.log")
        if (
            not stdout.exists()
            and not stdout.is_symlink()
            and not stderr.exists()
            and not stderr.is_symlink()
        ):
            return candidate
    raise ContractError("start-position validation exhausted its immutable recovery log paths")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
