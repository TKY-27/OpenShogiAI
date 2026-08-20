"""Bounded, resumable Phase 5 evaluator comparisons and evidence verification."""

from __future__ import annotations

import hashlib
import math
import os
import random
import stat
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final

from open_shogi_training.labeling.artifacts import ArtifactError, stable_regular_descriptor
from open_shogi_training.labeling.execution import ExecutableSnapshot, ExecutableSnapshotError
from open_shogi_training.models.export import (
    ARCH_VERSION,
    QUANTIZATION_FLOAT32,
    QUANTIZATION_INT8,
    parse_value_model,
)
from open_shogi_training.selfplay.arena import validate_phase2_pair_report
from open_shogi_training.selfplay.common import (
    ArtifactRef,
    ContractError,
    artifact_ref,
    canonical_json_bytes,
    contained_path,
    ensure_contained_directory,
    load_json,
    load_json_artifact,
    require_clean_head,
    write_json_new,
)
from open_shogi_training.selfplay.engine_receipt import (
    DEFAULT_ENGINE_PATH,
    resolve_active_engine_build,
)
from open_shogi_training.selfplay.planning import (
    StartPosition,
    validate_start_position_validation,
)
from open_shogi_training.selfplay.starts import validate_start_position_source_artifact

PHASE5_ARENA_SCHEMA: Final = "phase5_arena_manifest/v2"
PHASE5_GIT_COMMIT: Final = ""
PHASE5_PAIRS: Final = 20
PHASE5_GAMES_PER_PAIR: Final = 2
PHASE5_NODES: Final = 500
PHASE5_MAX_PLIES: Final = 128
PHASE5_DEPTH: Final = 4
PHASE5_HASH_MIB: Final = 16
PHASE5_BOOTSTRAP_SAMPLES: Final = 100_000
MAX_REPORT_BYTES: Final = 8 * 1024 * 1024

_COUNTER_KEYS = (
    "neuralInferenceCalls",
    "neuralInferenceTimeNs",
    "playerASearchNodes",
    "playerASearchElapsedMs",
    "playerADepthSum",
    "playerASearches",
    "playerANeuralInferenceCalls",
    "playerANeuralInferenceTimeNs",
    "playerBSearchNodes",
    "playerBSearchElapsedMs",
    "playerBDepthSum",
    "playerBSearches",
    "playerBNeuralInferenceCalls",
    "playerBNeuralInferenceTimeNs",
)


@dataclass(frozen=True, slots=True)
class Phase5Comparison:
    name: str
    player_a: str
    player_b: str
    model_a: Path | None
    model_b: Path | None
    player_b_opening: bool
    seed_index: int


def phase5_comparisons(*, f32_model: Path, int8_model: Path) -> tuple[Phase5Comparison, ...]:
    """Return the frozen comparison matrix in its original seed order."""

    return (
        Phase5Comparison(
            "material-vs-baseline", "material", "handcrafted-baseline", None, None, False, 0
        ),
        Phase5Comparison(
            "baseline-vs-experimental",
            "handcrafted-baseline",
            "handcrafted-experimental",
            None,
            None,
            False,
            1,
        ),
        Phase5Comparison(
            "experimental-vs-neural-f32",
            "handcrafted-experimental",
            "neural",
            None,
            f32_model,
            False,
            2,
        ),
        Phase5Comparison("neural-f32-vs-int8", "neural", "neural", f32_model, int8_model, False, 3),
        Phase5Comparison(
            "neural-int8-opening-off-vs-on",
            "neural",
            "neural",
            int8_model,
            int8_model,
            True,
            4,
        ),
    )


def run_phase5_arena(
    *,
    repository_root: Path,
    engine: Path,
    starts_path: Path,
    starts_validation_path: Path,
    output_root: Path,
    f32_model: Path,
    int8_model: Path,
    opening_book: Path,
    git_commit: str,
) -> None:
    """Run or safely resume the exact five-by-forty Phase 5 matrix."""

    require_clean_head(repository_root, git_commit)
    engine, engine_ref, build_receipt_ref = _validate_repository_engine(
        repository_root, engine, git_commit
    )
    starts, _, _, _, _ = _load_starts(
        repository_root,
        starts_path,
        starts_validation_path,
        engine_ref=engine_ref,
        build_receipt_ref=build_receipt_ref,
    )
    comparisons = phase5_comparisons(f32_model=f32_model, int8_model=int8_model)
    for comparison in comparisons:
        for position_index, position in enumerate(starts):
            seed = _pair_seed(comparison, position_index)
            pair_directory = (
                output_root
                / comparison.name
                / f"pair-{position_index + 1:02d}-{position.position_id[:12]}"
            )
            report_path = pair_directory / "arena-report.json"
            if report_path.exists():
                _verify_pair(
                    repository_root=repository_root,
                    engine=engine,
                    expected_engine=engine_ref,
                    report_path=report_path,
                    comparison=comparison,
                    position=position,
                    pair_index=position_index,
                    git_commit=git_commit,
                    opening_book=opening_book,
                    validate_csa=True,
                )
                continue
            if pair_directory.exists() and not (pair_directory / "arena.state").is_file():
                raise ContractError(
                    f"refusing a non-resumable existing Phase 5 pair directory: {pair_directory}"
                )
            command = [
                os.fspath(engine),
                "arena",
                "--games",
                str(PHASE5_GAMES_PER_PAIR),
                "--player-a",
                comparison.player_a,
                "--player-b",
                comparison.player_b,
                "--a-depth",
                str(PHASE5_DEPTH),
                "--b-depth",
                str(PHASE5_DEPTH),
                "--a-hash-mb",
                str(PHASE5_HASH_MIB),
                "--b-hash-mb",
                str(PHASE5_HASH_MIB),
                "--nodes",
                str(PHASE5_NODES),
                "--sfen",
                position.sfen,
                "--max-plies",
                str(PHASE5_MAX_PLIES),
                "--seed",
                str(seed),
                "--git-commit",
                git_commit,
                "--output-dir",
                _repository_relative(repository_root, pair_directory),
            ]
            if comparison.model_a is not None:
                command.extend(
                    ("--a-model", _repository_relative(repository_root, comparison.model_a))
                )
            if comparison.model_b is not None:
                command.extend(
                    ("--b-model", _repository_relative(repository_root, comparison.model_b))
                )
            if comparison.player_b_opening:
                command.extend(
                    (
                        "--b-opening",
                        "--opening-book",
                        _repository_relative(repository_root, opening_book),
                        "--opening-max-plies",
                        "24",
                    )
                )
            if (pair_directory / "arena.state").is_file():
                command.append("--resume")
            _run_engine_command(
                engine,
                command[1:],
                repository_root=repository_root,
                expected_engine=engine_ref,
                timeout_seconds=3_600,
                memory_limit_mib=2_048,
            )
            _verify_pair(
                repository_root=repository_root,
                engine=engine,
                expected_engine=engine_ref,
                report_path=report_path,
                comparison=comparison,
                position=position,
                pair_index=position_index,
                git_commit=git_commit,
                opening_book=opening_book,
                validate_csa=True,
            )


def verify_phase5_arena(
    *,
    repository_root: Path,
    engine: Path,
    starts_path: Path,
    starts_validation_path: Path,
    output_root: Path,
    manifest_path: Path,
    f32_model: Path,
    int8_model: Path,
    opening_book: Path,
    git_commit: str,
) -> dict[str, object]:
    """Recompute all Phase 5 results from immutable reports and Rust-replayed CSA."""

    require_clean_head(repository_root, git_commit)
    engine, engine_ref, build_receipt_ref = _validate_repository_engine(
        repository_root, engine, git_commit
    )
    starts, starts_ref, validation_ref, dataset_ref, source_positions_ref = _load_starts(
        repository_root,
        starts_path,
        starts_validation_path,
        engine_ref=engine_ref,
        build_receipt_ref=build_receipt_ref,
    )
    comparisons = sorted(
        phase5_comparisons(f32_model=f32_model, int8_model=int8_model),
        key=lambda item: item.name,
    )
    summaries: list[dict[str, object]] = []
    reports_verified = 0
    csa_verified = 0
    for bootstrap_index, comparison in enumerate(comparisons):
        accumulator = _ComparisonAccumulator()
        for pair_index, position in enumerate(starts):
            report_path = (
                output_root
                / comparison.name
                / f"pair-{pair_index + 1:02d}-{position.position_id[:12]}"
                / "arena-report.json"
            )
            report = _verify_pair(
                repository_root=repository_root,
                engine=engine,
                expected_engine=engine_ref,
                report_path=report_path,
                comparison=comparison,
                position=position,
                pair_index=pair_index,
                git_commit=git_commit,
                opening_book=opening_book,
                validate_csa=True,
            )
            accumulator.add(
                report=report,
                report_path=report_path,
                position=position,
                repository_root=repository_root,
            )
            reports_verified += 1
            csa_verified += 2
        summaries.append(accumulator.summary(comparison.name, bootstrap_index))
    manifest: dict[str, object] = {
        "schema": PHASE5_ARENA_SCHEMA,
        "engine": engine_ref.as_dict(),
        "engineBuildReceipt": build_receipt_ref.as_dict(),
        "startPositions": starts_ref.as_dict(),
        "startPositionValidation": validation_ref.as_dict(),
        "datasetManifest": dataset_ref.as_dict(),
        "sourcePositions": source_positions_ref.as_dict(),
        "contract": {
            "pairs": PHASE5_PAIRS,
            "gamesPerPair": PHASE5_GAMES_PER_PAIR,
            "nodesPerMove": PHASE5_NODES,
            "maxPlies": PHASE5_MAX_PLIES,
            "searchDepth": PHASE5_DEPTH,
            "hashMiB": PHASE5_HASH_MIB,
            "bootstrapSamples": PHASE5_BOOTSTRAP_SAMPLES,
        },
        "reportsVerified": reports_verified,
        "csaFilesVerified": csa_verified,
        "comparisons": summaries,
    }
    serialized = canonical_json_bytes(manifest)
    if manifest_path.exists():
        existing = _read_regular_bytes(manifest_path, MAX_REPORT_BYTES)
        if existing != serialized:
            raise ContractError("existing Phase 5 arena manifest differs from recomputed evidence")
    else:
        write_json_new(manifest_path, manifest)
    return manifest


def _load_starts(
    repository_root: Path,
    path: Path,
    validation_path: Path,
    *,
    engine_ref: ArtifactRef,
    build_receipt_ref: ArtifactRef,
) -> tuple[
    tuple[StartPosition, ...],
    ArtifactRef,
    ArtifactRef,
    ArtifactRef,
    ArtifactRef,
]:
    starts_ref = artifact_ref(
        repository_root,
        _repository_relative(repository_root, path),
        maximum_bytes=MAX_REPORT_BYTES,
    )
    starts_raw = load_json_artifact(repository_root, starts_ref)
    starts = validate_start_position_source_artifact(
        repository_root=repository_root,
        start_positions=starts_raw,
        start_positions_ref=starts_ref,
    )
    validation_ref = artifact_ref(
        repository_root,
        _repository_relative(repository_root, validation_path),
        maximum_bytes=MAX_REPORT_BYTES,
    )
    validate_start_position_validation(
        load_json_artifact(repository_root, validation_ref),
        start_positions_ref=starts_ref,
        engine_ref=engine_ref,
        positions=starts,
        repository_root=repository_root,
        engine_build_receipt=build_receipt_ref,
        runtime_authorization=True,
    )
    if len(starts.positions) != PHASE5_PAIRS:
        raise ContractError(f"Phase 5 requires exactly {PHASE5_PAIRS} distinct starts")
    if Counter(position.split for position in starts.positions) != Counter(
        {"train": 10, "validation": 10}
    ):
        raise ContractError("Phase 5 requires ten train and ten validation start positions")
    return (
        starts.positions,
        starts_ref,
        validation_ref,
        starts.dataset_manifest,
        starts.source_positions,
    )


def _pair_seed(comparison: Phase5Comparison, pair_index: int) -> int:
    return 20_260_810 + comparison.seed_index * 100 + pair_index


def _verify_pair(
    *,
    repository_root: Path,
    engine: Path,
    expected_engine: ArtifactRef,
    report_path: Path,
    comparison: Phase5Comparison,
    position: StartPosition,
    pair_index: int,
    git_commit: str,
    opening_book: Path,
    validate_csa: bool,
) -> dict[str, Any]:
    report = dict(
        validate_phase2_pair_report(
            load_json(report_path, maximum_bytes=MAX_REPORT_BYTES), context=str(report_path)
        )
    )
    run = report["run"]
    expected = {
        "seed": _pair_seed(comparison, pair_index),
        "gameLimit": PHASE5_GAMES_PER_PAIR,
        "gitCommit": git_commit,
        "initialSfen": position.sfen,
        "maxPlies": PHASE5_MAX_PLIES,
        "budget": {"kind": "nodes", "value": PHASE5_NODES},
    }
    if any(run[key] != value for key, value in expected.items()) or run["completedAt"] is None:
        raise ContractError(f"Phase 5 report differs from its frozen run contract: {report_path}")
    expected_a = _expected_player(comparison.player_a, comparison.model_a, opening=False)
    expected_b = _expected_player(
        comparison.player_b, comparison.model_b, opening=comparison.player_b_opening
    )
    if run["playerA"] != expected_a or run["playerB"] != expected_b:
        raise ContractError(f"Phase 5 report player identity drifted: {report_path}")
    if comparison.player_b_opening:
        opening_bytes = _read_regular_bytes(opening_book, 512 * 1024 * 1024)
        expected_opening = {
            "enabled": True,
            "artifactSha256": hashlib.sha256(opening_bytes).hexdigest(),
            "artifactSize": len(opening_bytes),
            "maxPlies": 24,
        }
    else:
        expected_opening = {
            "enabled": False,
            "artifactSha256": None,
            "artifactSize": None,
            "maxPlies": None,
        }
    if run["opening"] != expected_opening:
        raise ContractError(f"Phase 5 report opening identity drifted: {report_path}")
    pair_directory = report_path.parent
    for game in report["games"]:
        relative = PurePosixPath(game["csaPath"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ContractError(f"unsafe CSA path in {report_path}")
        csa_path = pair_directory.joinpath(*relative.parts)
        data = _read_regular_bytes(csa_path, 8 * 1024 * 1024)
        if len(data) != game["csaSize"] or hashlib.sha256(data).hexdigest() != game["csaSha256"]:
            raise ContractError(f"CSA identity mismatch: {csa_path}")
        if validate_csa:
            _run_engine_command(
                engine,
                ["validate-csa", os.fspath(csa_path)],
                repository_root=repository_root,
                expected_engine=expected_engine,
                timeout_seconds=120,
                memory_limit_mib=1_024,
            )
    return report


def _expected_player(
    evaluator: str, model_path: Path | None, *, opening: bool
) -> dict[str, object]:
    model_fields: dict[str, object] = {
        "modelArtifactSha256": None,
        "modelArtifactSize": None,
        "modelPayloadSha256": None,
        "architectureVersion": None,
        "quantization": None,
    }
    suffix = ""
    if evaluator == "neural":
        if model_path is None:
            raise ContractError("neural Phase 5 player lacks a model")
        artifact = _read_regular_bytes(model_path, 64 * 1024 * 1024)
        model = parse_value_model(artifact)
        artifact_sha = hashlib.sha256(artifact).hexdigest()
        quantization = {
            QUANTIZATION_FLOAT32: "float32",
            QUANTIZATION_INT8: "int8",
        }.get(model.quantization)
        if quantization is None:
            raise ContractError("Phase 5 model uses unsupported quantization")
        model_fields = {
            "modelArtifactSha256": artifact_sha,
            "modelArtifactSize": len(artifact),
            "modelPayloadSha256": model.payload_sha256,
            "architectureVersion": ARCH_VERSION,
            "quantization": quantization,
        }
        suffix = f":m-{artifact_sha[:12]}"
    elif model_path is not None:
        raise ContractError("non-neural Phase 5 player unexpectedly has a model")
    return {
        "label": (
            f"search:{evaluator}:d{PHASE5_DEPTH}:h{PHASE5_HASH_MIB}:"
            f"tt-on:book-{'on' if opening else 'off'}{suffix}"
        ),
        "evaluatorKind": evaluator,
        "searchDepth": PHASE5_DEPTH,
        "hashMegabytes": PHASE5_HASH_MIB,
        "transposition": True,
        **model_fields,
        "openingEnabled": opening,
    }


class _ComparisonAccumulator:
    def __init__(self) -> None:
        self.outcomes: Counter[str] = Counter()
        self.side_outcomes: Counter[str] = Counter()
        self.split_outcomes: dict[str, Counter[str]] = {
            "train": Counter(),
            "validation": Counter(),
        }
        self.pair_scores: list[float] = []
        self.moves: list[int] = []
        self.totals: Counter[str] = Counter()
        self.duration_seconds = 0.0
        self.opening_moves = 0
        self.labels: tuple[str, str] | None = None
        self.reports: list[dict[str, object]] = []

    def add(
        self,
        *,
        report: dict[str, Any],
        report_path: Path,
        position: StartPosition,
        repository_root: Path,
    ) -> None:
        run = report["run"]
        labels = (run["playerA"]["label"], run["playerB"]["label"])
        if self.labels is None:
            self.labels = labels
        elif self.labels != labels:
            raise ContractError("Phase 5 comparison changes player identity")
        self.duration_seconds += (_utc(run["completedAt"]) - _utc(run["startedAt"])).total_seconds()
        pair_score = 0.0
        initial_side = run["initialSfen"].split(" ")[1]
        for game in report["games"]:
            result = game["result"]
            self.moves.append(game["moves"])
            self.outcomes[result] += 1
            self.split_outcomes[position.split][result] += 1
            if result in {"draw", "max_plies"}:
                pair_score += 0.5
            else:
                winner = game["black"] if result == "black_win" else game["white"]
                side = "Black" if result == "black_win" else "White"
                logical = "A" if winner == labels[0] else "B"
                self.side_outcomes[f"player{logical}As{side}Wins"] += 1
                self.split_outcomes[position.split][f"player{logical}Wins"] += 1
                if logical == "A":
                    pair_score += 1.0
            for key in _COUNTER_KEYS:
                self.totals[key] += game[key]
            black_turns = (game["moves"] + int(initial_side == "b")) // 2
            white_turns = game["moves"] - black_turns
            player_b_turns = black_turns if game["black"] == labels[1] else white_turns
            if game["playerBSearches"] > player_b_turns:
                raise ContractError("Phase 5 player B search count exceeds its turns")
            self.opening_moves += player_b_turns - game["playerBSearches"]
        self.pair_scores.append(pair_score / 2)
        self.reports.append(_artifact_ref(repository_root, report_path))

    def summary(self, name: str, bootstrap_index: int) -> dict[str, object]:
        if self.labels is None or len(self.moves) != PHASE5_PAIRS * PHASE5_GAMES_PER_PAIR:
            raise ContractError(f"incomplete Phase 5 comparison: {name}")
        a_wins = self.side_outcomes["playerAAsBlackWins"] + self.side_outcomes["playerAAsWhiteWins"]
        b_wins = self.side_outcomes["playerBAsBlackWins"] + self.side_outcomes["playerBAsWhiteWins"]
        draws = self.outcomes["draw"]
        max_plies = self.outcomes["max_plies"]
        games = len(self.moves)
        score = (a_wins + 0.5 * (draws + max_plies)) / games
        bootstrap = _paired_bootstrap(self.pair_scores, 20_260_813 + bootstrap_index)
        searches = self.totals["playerASearches"] + self.totals["playerBSearches"]
        nodes = self.totals["playerASearchNodes"] + self.totals["playerBSearchNodes"]
        elapsed = self.totals["playerASearchElapsedMs"] + self.totals["playerBSearchElapsedMs"]
        depth = self.totals["playerADepthSum"] + self.totals["playerBDepthSum"]
        decisive = a_wins + b_wins
        return {
            "name": name,
            "playerA": self.labels[0],
            "playerB": self.labels[1],
            "games": games,
            "playerAWins": a_wins,
            "playerBWins": b_wins,
            "draws": draws,
            "maxPlies": max_plies,
            "blackWins": self.outcomes["black_win"],
            "whiteWins": self.outcomes["white_win"],
            "playerAScoreRateTreatingMaxPliesAsHalf": score,
            "pairedBootstrap95": bootstrap,
            "decisivePlayerAWinRate": a_wins / decisive if decisive else None,
            "decisivePlayerAWilson95": _wilson(a_wins, decisive),
            "eloApproximation": _elo(score),
            "eloApproximationBootstrap95": [_elo(bootstrap[0]), _elo(bootstrap[1])],
            "playerAWinsAsBlack": self.side_outcomes["playerAAsBlackWins"],
            "playerAWinsAsWhite": self.side_outcomes["playerAAsWhiteWins"],
            "playerBWinsAsBlack": self.side_outcomes["playerBAsBlackWins"],
            "playerBWinsAsWhite": self.side_outcomes["playerBAsWhiteWins"],
            "averageMoves": statistics.fmean(self.moves),
            "averageNodesPerSearchedMove": _safe_ratio(nodes, searches),
            "averageDepth": _safe_ratio(depth, searches),
            "averageSearchMilliseconds": _safe_ratio(elapsed, searches),
            "nodesPerSecond": _safe_ratio(nodes * 1000, elapsed),
            "neuralInferenceCalls": self.totals["neuralInferenceCalls"],
            "neuralInferenceTimeNs": self.totals["neuralInferenceTimeNs"],
            "openingMovesByPlayerB": self.opening_moves,
            "illegalMoves": 0,
            "crashes": 0,
            "sumPairWallClockSeconds": self.duration_seconds,
            "splitOutcomes": {key: dict(value) for key, value in self.split_outcomes.items()},
            "reports": self.reports,
        }


def _safe_ratio(numerator: int, denominator: int) -> float:
    """Return a finite, schema-stable rate for an empty measurement bucket."""

    return numerator / denominator if denominator else 0.0


def _artifact_ref(repository_root: Path, path: Path) -> dict[str, object]:
    data = _read_regular_bytes(path, MAX_REPORT_BYTES)
    return {
        "path": _repository_relative(repository_root, path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
    }


def _repository_relative(root: Path, path: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(root.resolve(strict=True)).as_posix()
    except ValueError as error:
        raise ContractError(f"path is outside the repository: {path}") from error


def _regular_file(path: Path, context: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ContractError(f"{context} is unavailable: {path}") from error
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ContractError(f"{context} must be a regular non-symlink file: {path}")
    return path


def _read_regular_bytes(path: Path, maximum: int) -> bytes:
    try:
        with stable_regular_descriptor(path) as descriptor:
            before = os.fstat(descriptor)
            if not 0 < before.st_size <= maximum:
                raise ContractError(f"artifact violates its size bound: {path}")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                data = stream.read(maximum + 1)
                after = os.fstat(stream.fileno())
    except ArtifactError as error:
        raise ContractError(f"artifact is not a stable regular file: {path}") from error
    if len(data) > maximum or (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ContractError(f"artifact changed while being read: {path}")
    return data


def _validate_repository_engine(
    repository_root: Path,
    engine: Path,
    git_commit: str,
) -> tuple[Path, ArtifactRef, ArtifactRef]:
    relative = _repository_relative(repository_root, engine)
    if relative != DEFAULT_ENGINE_PATH:
        raise ContractError("Phase 5 engine option must remain the operator alias")
    reference, receipt, _ = resolve_active_engine_build(
        repository_root, expected_git_commit=git_commit
    )
    immutable = contained_path(repository_root, reference.path, must_exist=True)
    return immutable, reference, receipt


def _run_engine_command(
    engine: Path,
    arguments: list[str],
    *,
    repository_root: Path,
    expected_engine: ArtifactRef,
    timeout_seconds: int,
    memory_limit_mib: int,
) -> None:
    from open_shogi_training.selfplay.execution import _run_bounded_process

    snapshot_root = ensure_contained_directory(repository_root, "local/runtime-snapshots")
    observed = artifact_ref(
        repository_root,
        _repository_relative(repository_root, engine),
        maximum_bytes=512 * 1024 * 1024,
    )
    if observed != expected_engine:
        raise ContractError("Phase 5 engine differs from its validated build receipt")
    try:
        snapshot = ExecutableSnapshot.create(
            engine,
            temporary_directory=snapshot_root,
            max_bytes=512 * 1024 * 1024,
            expected_sha256=expected_engine.sha256,
        )
    except ExecutableSnapshotError as error:
        raise ContractError(f"cannot snapshot Phase 5 engine: {error}") from error
    try:
        (
            _,
            stderr,
            return_code,
            timed_out,
            output_limit,
            memory_limit,
            _,
            _,
        ) = _run_bounded_process(
            [snapshot.executable_path, *arguments],
            cwd=repository_root,
            timeout_seconds=timeout_seconds,
            pass_fds=snapshot.pass_fds(),
            memory_limit_bytes=memory_limit_mib * 1024 * 1024,
            launch_guard=lambda: (
                snapshot.assert_snapshot_unchanged(),
                snapshot.assert_source_unchanged(),
            ),
        )
        snapshot.assert_snapshot_unchanged()
        snapshot.assert_source_unchanged()
    except ExecutableSnapshotError as error:
        raise ContractError(f"Phase 5 engine snapshot changed: {error}") from error
    finally:
        snapshot.close()
    if return_code != 0 or timed_out or output_limit or memory_limit:
        detail = stderr[-2_048:].decode("utf-8", errors="replace")
        raise ContractError(f"Phase 5 engine command failed safely: {detail}")


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _wilson(wins: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    proportion = wins / total
    denominator = 1 + z * z / total
    center = proportion + z * z / (2 * total)
    margin = z * math.sqrt((proportion * (1 - proportion) + z * z / (4 * total)) / total)
    return [
        max(0.0, (center - margin) / denominator),
        min(1.0, (center + margin) / denominator),
    ]


def _paired_bootstrap(pair_scores: list[float], seed: int) -> list[float]:
    random_source = random.Random(seed)
    count = len(pair_scores)
    samples = sorted(
        sum(random_source.choice(pair_scores) for _ in range(count)) / count
        for _ in range(PHASE5_BOOTSTRAP_SAMPLES)
    )
    return [samples[2_500], samples[97_499]]


def _elo(score: float) -> float | None:
    return 400 * math.log10(score / (1 - score)) if 0 < score < 1 else None
