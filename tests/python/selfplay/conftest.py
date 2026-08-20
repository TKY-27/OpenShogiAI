from __future__ import annotations

import hashlib
import os
import struct
import subprocess
import sys
import time
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from open_shogi_training.data.aobazero import adapt_aobazero_csa
from open_shogi_training.data.gzip_jsonl import write_jsonl_gzip_atomic
from open_shogi_training.data.registry import load_source_registry
from open_shogi_training.data.splits import SplitPolicy, assign_game_split
from open_shogi_training.selfplay.arena import (
    arena_config_signature_bytes,
    collect_arena_results,
)
from open_shogi_training.selfplay.common import (
    ArtifactRef,
    canonical_json_bytes,
    canonical_sha256,
)
from open_shogi_training.selfplay.config import SelfPlayConfig, load_selfplay_config
from open_shogi_training.selfplay.planning import (
    ModelSpec,
    StartPositionSet,
    build_arena_plan,
    parse_start_positions,
    start_position_identity,
)
from open_shogi_training.selfplay.registry import (
    build_initial_registry,
    register_challenger_generation,
)
from open_shogi_training.selfplay.starts import build_start_positions_from_phase3

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@contextmanager
def held_directory_authority(directory: Path):
    """Hold a real cross-process directory lock until the parent releases it."""

    directory.parent.mkdir(parents=True, exist_ok=True)
    ready = directory.parent / f".{directory.name}.lock-test-ready"
    release = directory.parent / f".{directory.name}.lock-test-release"
    program = """
import sys
import time
from pathlib import Path
from open_shogi_training.labeling.artifacts import stable_directory_lock

directory = Path(sys.argv[1])
ready = Path(sys.argv[2])
release = Path(sys.argv[3])
with stable_directory_lock(directory, create=True, exclusive=True, nonblocking=True):
    ready.write_bytes(b"ready\\n")
    deadline = time.monotonic() + 10.0
    while not release.exists():
        if time.monotonic() >= deadline:
            raise SystemExit("lock-test release timed out")
        time.sleep(0.01)
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.fspath(PROJECT_ROOT / "training")
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            program,
            os.fspath(directory),
            os.fspath(ready),
            os.fspath(release),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 5.0
    while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    if not ready.exists():
        stdout, stderr = process.communicate(timeout=2)
        raise AssertionError(
            "lock holder did not become ready: "
            f"stdout={stdout!r}, stderr={stderr!r}, returncode={process.returncode}"
        )
    try:
        yield
    finally:
        release.write_bytes(b"release\n")
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate(timeout=2)
            raise AssertionError(
                f"lock holder did not stop: stdout={stdout!r}, stderr={stderr!r}"
            ) from None
        if process.returncode != 0:
            raise AssertionError(
                "lock holder failed: "
                f"stdout={stdout!r}, stderr={stderr!r}, returncode={process.returncode}"
            )


@pytest.fixture
def selfplay_config() -> SelfPlayConfig:
    return load_selfplay_config(PROJECT_ROOT / "configs" / "selfplay" / "phase6_smoke.toml")


def make_ref(path: str, contents: bytes = b"fixture\n") -> ArtifactRef:
    return ArtifactRef(
        path=path,
        sha256=hashlib.sha256(contents).hexdigest(),
        size=len(contents),
    )


def write_ref(root: Path, path: str, contents: bytes = b"fixture\n") -> ArtifactRef:
    destination = root / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(contents)
    return make_ref(path, contents)


def write_completed_attempt_receipt(
    root: Path,
    *,
    plan: Mapping[str, Any],
    job: Mapping[str, Any],
    stdout: ArtifactRef,
    stderr: ArtifactRef,
    report: ArtifactRef,
    csa: tuple[ArtifactRef, ArtifactRef],
    completed_at: str = "2026-08-08T00:00:01Z",
) -> ArtifactRef:
    """Write the exact process -> attempt receipt chain for one fixture job."""

    output_dir = str(job["outputDir"])
    run_root = output_dir.rsplit("/jobs/", 1)[0]
    receipt_root = f"{run_root}/logs/{job['jobId']}"
    invocation = {
        "command": job["command"],
        "resume": False,
        "memoryLimitMiB": plan["memoryPerWorkerMiB"],
        "expectedExecutable": plan["engine"],
        "engineBuildReceipt": plan.get("engineBuildReceipt"),
        "runtimeReceipt": None,
    }
    result = {
        "returnCode": 0,
        "timedOut": False,
        "outputLimitExceeded": False,
        "memoryLimitExceeded": False,
        "peakRssBytes": 1,
        "rssMeasurement": "process_tree_ps_rss_sum",
    }
    process = {
        "schema": "phase6_command_receipt/v2",
        **invocation,
        "commandSha256": canonical_sha256(invocation),
        **result,
        "stdout": stdout.as_dict(),
        "stderr": stderr.as_dict(),
    }
    process_ref = write_ref(
        root,
        f"{receipt_root}/attempt-001.receipt.json",
        canonical_json_bytes(process),
    )
    evidence: dict[str, object] = {
        "schema": "phase6_attempt_command_receipt/v1",
        "planSha256": plan["planSha256"],
        "jobId": job["jobId"],
        "attempt": 1,
        "command": job["command"],
        "engine": plan["engine"],
        "engineBuildReceipt": plan.get("engineBuildReceipt"),
        "processReceipt": process_ref.as_dict(),
        "result": result,
        "stdout": stdout.as_dict(),
        "stderr": stderr.as_dict(),
        "report": report.as_dict(),
        "csa": [item.as_dict() for item in csa],
        "quarantine": None,
        "failureCategory": None,
        "completedAt": completed_at,
    }
    evidence["receiptSha256"] = canonical_sha256(evidence)
    return write_ref(
        root,
        f"{receipt_root}/attempt-001.evidence.json",
        canonical_json_bytes(evidence),
    )


def write_selfplay_config_ref(root: Path) -> ArtifactRef:
    return write_ref(
        root,
        "configs/selfplay.toml",
        (PROJECT_ROOT / "configs/selfplay/phase6_smoke.toml").read_bytes(),
    )


def write_initial_registry_ref(
    root: Path,
    *,
    champion_ref: ArtifactRef,
    quantization: str,
) -> ArtifactRef:
    registry = build_initial_registry(
        generation_id="generation-0",
        champion_model_id="champion-v0",
        champion_artifact=champion_ref,
        evaluator_kind="neural",
        architecture_version="1",
        quantization=quantization,
        training_run=None,
        registered_at="2026-08-08T00:00:00Z",
    )
    return write_ref(
        root,
        "weights/model-registry.json",
        canonical_json_bytes(registry),
    )


def write_engine_build_receipt_ref(
    root: Path,
    *,
    engine_ref: ArtifactRef,
    git_commit: str = "a399407",
) -> ArtifactRef:
    """Write closed-valid immutable v2 build evidence without authorizing execution."""

    full_commit = (
        "a3994070a4d4d93645e54b19281c45a42bd46f2a"
        if git_commit == "a399407"
        else git_commit.ljust(40, "0")
    )
    receipt: dict[str, object] = {
        "schema": "open_shogi_engine_build_receipt/v2",
        "gitCommit": full_commit,
        "binary": engine_ref.as_dict(),
        "sourceTreeSha256": hashlib.sha256(b"fixture-engine-source-tree").hexdigest(),
        "sourceFiles": 1,
        "sourceBytes": 1,
        "buildCommand": [
            "cargo",
            "build",
            "--locked",
            "--release",
            "--offline",
            "--jobs",
            "4",
            "-p",
            "open-shogi-cli",
        ],
        "cargoTool": {
            "path": "/fixture/toolchain/bin/cargo",
            "sha256": hashlib.sha256(b"fixture-cargo").hexdigest(),
            "size": len(b"fixture-cargo"),
            "version": "cargo fixture",
        },
        "rustcTool": {
            "path": "/fixture/toolchain/bin/rustc",
            "sha256": hashlib.sha256(b"fixture-rustc").hexdigest(),
            "size": len(b"fixture-rustc"),
            "version": "rustc fixture",
        },
        "rustcRuntimeTree": {
            "treeSha256": hashlib.sha256(b"fixture-rustc-runtime").hexdigest(),
            "files": 2,
            "bytes": 2,
        },
    }
    receipt["receiptSha256"] = canonical_sha256(receipt)
    return write_ref(
        root,
        f"local/build-receipts/open-shogi-cli/{receipt['receiptSha256']}.json",
        canonical_json_bytes(receipt),
    )


def write_content_addressed_engine_ref(root: Path) -> ArtifactRef:
    """Write the immutable engine identity used by durable Phase 6 fixtures."""

    contents = b"\xcf\xfa\xed\xfe" + b"fixture-engine" * 8
    sha256 = hashlib.sha256(contents).hexdigest()
    return write_ref(
        root,
        f"local/builds/open-shogi-cli/{sha256}/open-shogi-cli",
        contents,
    )


def write_arena_registry_ref(
    root: Path,
    *,
    champion_ref: ArtifactRef,
    champion_quantization: str,
    challenger_ref: ArtifactRef,
    challenger_quantization: str,
) -> ArtifactRef:
    registry = build_initial_registry(
        generation_id="generation-0",
        champion_model_id="champion-v0",
        champion_artifact=champion_ref,
        evaluator_kind="neural",
        architecture_version="1",
        quantization=champion_quantization,
        training_run=None,
        registered_at="2026-08-08T00:00:00Z",
    )
    selfplay = write_ref(root, "artifacts/registry/selfplay.json", b"selfplay")
    labels = write_ref(root, "artifacts/registry/labels.json", b"labels")
    training = write_ref(root, "artifacts/registry/training.json", b"training")
    registry = register_challenger_generation(
        registry,
        repository_root=root,
        generation_id="generation-0001",
        parent_generation_id="generation-0",
        challenger_model_id="challenger-v1",
        challenger_artifact=challenger_ref,
        architecture_version="1",
        quantization=challenger_quantization,
        selfplay_manifest=selfplay,
        teacher_labeling_manifest=labels,
        training_run_manifest=training,
        registered_at="2026-08-08T01:00:00Z",
    )
    return write_ref(
        root,
        "weights/model-registry.json",
        canonical_json_bytes(registry),
    )


def tiny_model_bytes(*, quantization: int = 0) -> bytes:
    """Return one valid, dependency-free OSAVAL01 test model."""

    header = struct.pack(
        "<8s10If",
        b"OSAVAL01",
        1,
        1,
        1,
        1 << 2,
        1,
        1,
        1,
        0,
        quantization,
        2,
        1_200.0,
    )
    if quantization == 0:
        first = struct.pack("<IIff", 1, 1, 2.0, 3.0)
        second = struct.pack("<IIff", 1, 1, 4.0, 5.0)
    else:
        first = struct.pack("<IIfbf", 1, 1, 0.5, 4, 3.0)
        second = struct.pack("<IIfbf", 1, 1, 1.0, 4, 5.0)
    payload = header + first + second
    return payload + hashlib.sha256(payload).digest()


def make_phase2_pair_report(
    *,
    job: Mapping[str, Any],
    model_a_ref: ArtifactRef,
    model_a_bytes: bytes,
    model_b_ref: ArtifactRef,
    model_b_bytes: bytes,
    csa_refs: tuple[ArtifactRef, ArtifactRef],
    results: tuple[str, str] = ("black_win", "white_win"),
    replay_counters: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None,
    git_commit: str = "a399407",
) -> dict[str, object]:
    """Build one counter-consistent Rust v2 pair report for boundary tests."""

    player_a = _report_player(model_a_ref, model_a_bytes)
    player_b = _report_player(model_b_ref, model_b_bytes)
    games: list[dict[str, object]] = []
    for index, (result, csa_ref) in enumerate(zip(results, csa_refs, strict=True)):
        if replay_counters is None:
            moves = 80 + index * 10
            player_a_searches = 2
            player_b_searches = 2
            player_a_nodes = 100
            player_b_nodes = 80
            player_a_elapsed = 10
            player_b_elapsed = 10
            player_a_depth = 12
            player_b_depth = 10
            player_a_inferences = 2
            player_b_inferences = 3
            player_a_inference_time = 200
            player_b_inference_time = 300
        else:
            moves, player_a_searches, player_b_searches = replay_counters[index]
            player_a_nodes = player_a_searches
            player_b_nodes = player_b_searches
            player_a_elapsed = 0
            player_b_elapsed = 0
            player_a_depth = player_a_searches
            player_b_depth = player_b_searches
            player_a_inferences = 0
            player_b_inferences = 0
            player_a_inference_time = 0
            player_b_inference_time = 0
        games.append(
            {
                "id": index,
                "black": player_a["label"] if index == 0 else player_b["label"],
                "white": player_b["label"] if index == 0 else player_a["label"],
                "result": result,
                "moves": moves,
                "csaPath": f"games/game-{index + 1:06d}.csa",
                "csaSha256": csa_ref.sha256,
                "csaSize": csa_ref.size,
                "neuralInferenceCalls": player_a_inferences + player_b_inferences,
                "neuralInferenceTimeNs": (player_a_inference_time + player_b_inference_time),
                "playerASearchNodes": player_a_nodes,
                "playerASearchElapsedMs": player_a_elapsed,
                "playerADepthSum": player_a_depth,
                "playerASearches": player_a_searches,
                "playerANeuralInferenceCalls": player_a_inferences,
                "playerANeuralInferenceTimeNs": player_a_inference_time,
                "playerBSearchNodes": player_b_nodes,
                "playerBSearchElapsedMs": player_b_elapsed,
                "playerBDepthSum": player_b_depth,
                "playerBSearches": player_b_searches,
                "playerBNeuralInferenceCalls": player_b_inferences,
                "playerBNeuralInferenceTimeNs": player_b_inference_time,
            }
        )
    player_a_wins = sum(
        result == ("black_win" if index == 0 else "white_win")
        for index, result in enumerate(results)
    )
    player_b_wins = sum(
        result == ("white_win" if index == 0 else "black_win")
        for index, result in enumerate(results)
    )
    decisive = sum(result in {"black_win", "white_win"} for result in results)
    metric_names = (
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
    totals = {name: sum(int(game[name]) for game in games) for name in metric_names}
    total_nodes = totals["playerASearchNodes"] + totals["playerBSearchNodes"]
    total_elapsed = totals["playerASearchElapsedMs"] + totals["playerBSearchElapsedMs"]
    total_depth = totals["playerADepthSum"] + totals["playerBDepthSum"]
    total_searches = totals["playerASearches"] + totals["playerBSearches"]
    report: dict[str, object] = {
        "schema": "phase2_arena_report/v2",
        "run": {
            "seed": job["seed"],
            "gameLimit": 2,
            "engine": (
                f"OpenShogiAI 0.0.0 a={player_a['label']} b={player_b['label']} budget=Nodes(500)"
            ),
            "gitCommit": git_commit,
            "startedAt": "2026-08-08T00:00:00Z",
            "completedAt": "2026-08-08T00:00:01Z",
            "initialSfen": job["sfen"],
            "maxPlies": 256,
            "configSha256": "0" * 64,
            "budget": {"kind": "nodes", "value": 500},
            "playerA": player_a,
            "playerB": player_b,
            "opening": {
                "enabled": False,
                "artifactSha256": None,
                "artifactSize": None,
                "maxPlies": None,
            },
        },
        "metrics": {
            "games": 2,
            "finishedGames": sum(result != "max_plies" for result in results),
            "playerAWins": player_a_wins,
            "playerBWins": player_b_wins,
            "searchWins": decisive,
            "draws": results.count("draw"),
            "nodesPerSecond": 0.0 if total_elapsed == 0 else total_nodes * 1_000 / total_elapsed,
            "averageDepth": 0.0 if total_searches == 0 else total_depth / total_searches,
            "ttHitRate": 0.0 if replay_counters is not None else 0.1,
            "cutoffRate": 0.0 if replay_counters is not None else 0.2,
            "pruningRate": 0.0 if replay_counters is not None else 0.3,
            "millisecondsPerMove": (0.0 if total_searches == 0 else total_elapsed / total_searches),
            **totals,
            "peakMemoryBytes": None,
            "illegalMoves": 0,
        },
        "games": games,
    }
    report_run = report["run"]
    assert isinstance(report_run, dict)
    report_run["configSha256"] = hashlib.sha256(
        arena_config_signature_bytes(report_run)
    ).hexdigest()
    return report


def write_complete_arena_evidence(
    root: Path,
    *,
    registry: Mapping[str, Any],
    champion_ref: ArtifactRef,
    champion_bytes: bytes,
    challenger_ref: ArtifactRef,
    challenger_bytes: bytes,
    outcome: str = "balanced",
) -> tuple[dict[str, object], ArtifactRef]:
    """Write a closed plan/execution/report/CSA chain and collect exact results."""

    if outcome not in {"challenger", "champion", "balanced"}:
        raise ValueError("unsupported arena fixture outcome")
    engine = write_content_addressed_engine_ref(root)
    engine_receipt = write_engine_build_receipt_ref(root, engine_ref=engine)
    starts, starts_ref, validation_ref, _, _ = write_validated_phase3_starts(
        root,
        engine_ref=engine,
        engine_build_receipt=engine_receipt,
        train_sfen=("lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 1"),
        validation_sfen=("lnsgkgsnl/1r5b1/ppppppppp/9/9/6P2/PPPPPP1PP/1B5R1/LNSGKGSNL w - 1"),
    )
    registry_ref = write_ref(
        root,
        "weights/model-registry.json",
        canonical_json_bytes(registry),
    )
    config_ref = write_selfplay_config_ref(root)
    config = load_selfplay_config(PROJECT_ROOT / "configs/selfplay/phase6_smoke.toml")
    plan = build_arena_plan(
        generation_id="generation-0001",
        champion=ModelSpec("champion-v0", champion_ref, "neural"),
        challenger=ModelSpec("challenger-v1", challenger_ref, "neural"),
        engine=engine,
        model_registry=registry_ref,
        git_commit="a399407",
        config=config,
        config_ref=config_ref,
        start_positions=starts,
        start_positions_ref=starts_ref,
        validation_ref=validation_ref,
        engine_build_receipt=engine_receipt,
    )
    plan_ref = write_ref(root, "artifacts/arena-plan.json", canonical_json_bytes(plan))
    player_a = _report_player(challenger_ref, challenger_bytes)
    player_b = _report_player(champion_ref, champion_bytes)
    start_positions_by_id = {position.position_id: position for position in starts.positions}
    _, phase3_positions = _phase3_fixture_games()
    source_moves = {
        (str(row["canonicalSha256"]), int(row["positionIndex"])): str(row["moveUsi"])
        for row in phase3_positions
        if row["moveUsi"] is not None
    }
    attempts: list[dict[str, object]] = []
    for job in plan["jobs"]:
        output_dir = str(job["outputDir"])
        report_results = {
            "challenger": ("black_win", "white_win"),
            "champion": ("white_win", "black_win"),
            "balanced": ("black_win", "black_win"),
        }[outcome]
        if job["startGroup"] == "initial":
            source_move = "9g9f"
        else:
            position = start_positions_by_id[str(job["startPositionId"])]
            source_move = source_moves[(position.source_game_sha256, position.position_index)]
        csa_references: list[ArtifactRef] = []
        replay_counters: list[tuple[int, int, int]] = []
        for index, result in enumerate(report_results):
            black = str(player_a["label"] if index == 0 else player_b["label"])
            white = str(player_b["label"] if index == 0 else player_a["label"])
            csa_bytes, moves, black_searches, white_searches = _arena_fixture_csa(
                sfen=str(job["sfen"]),
                black=black,
                white=white,
                result=result,
                source_move=source_move,
            )
            csa_references.append(
                write_ref(
                    root,
                    f"{output_dir}/games/game-{index + 1:06d}.csa",
                    csa_bytes,
                )
            )
            if index == 0:
                replay_counters.append((moves, black_searches, white_searches))
            else:
                replay_counters.append((moves, white_searches, black_searches))
        csa_refs = (csa_references[0], csa_references[1])
        report = make_phase2_pair_report(
            job=job,
            model_a_ref=challenger_ref,
            model_a_bytes=challenger_bytes,
            model_b_ref=champion_ref,
            model_b_bytes=champion_bytes,
            csa_refs=csa_refs,
            results=report_results,
            replay_counters=(replay_counters[0], replay_counters[1]),
        )
        report_ref = write_ref(
            root,
            f"{output_dir}/arena-report.json",
            canonical_json_bytes(report),
        )
        stdout_ref = write_ref(root, f"artifacts/logs/{job['jobId']}.stdout", b"ok\n")
        stderr_ref = write_ref(root, f"artifacts/logs/{job['jobId']}.stderr", b"")
        command_receipt = write_completed_attempt_receipt(
            root,
            plan=plan,
            job=job,
            stdout=stdout_ref,
            stderr=stderr_ref,
            report=report_ref,
            csa=csa_refs,
        )
        attempts.append(
            {
                "jobId": job["jobId"],
                "attempt": 1,
                "status": "completed",
                "returnCode": 0,
                "timedOut": False,
                "outputLimitExceeded": False,
                "memoryLimitExceeded": False,
                "peakRssBytes": 1,
                "rssMeasurement": "process_tree_ps_rss_sum",
                "stdout": stdout_ref.as_dict(),
                "stderr": stderr_ref.as_dict(),
                "report": report_ref.as_dict(),
                "csa": [reference.as_dict() for reference in csa_refs],
                "quarantine": None,
                "failureCategory": None,
                "completedAt": "2026-08-08T00:00:01Z",
                "commandReceipt": command_receipt.as_dict(),
            }
        )
    execution: dict[str, object] = {
        "schema": "phase6_arena_execution_manifest/v1",
        "generationId": "generation-0001",
        "plan": plan_ref.as_dict(),
        "planSha256": plan["planSha256"],
        "status": "completed",
        "gameCountPlanned": len(plan["jobs"]) * 2,
        "jobsPlanned": len(plan["jobs"]),
        "jobsCompleted": len(plan["jobs"]),
        "jobsQuarantined": 0,
        "gamesCompleted": len(plan["jobs"]) * 2,
        "gamesQuarantined": 0,
        "quarantinedAttempts": 0,
        "attempts": attempts,
    }
    execution["manifestSha256"] = canonical_sha256(execution)
    execution_ref = write_ref(
        root,
        "artifacts/arena-execution.json",
        canonical_json_bytes(execution),
    )
    results = collect_arena_results(
        repository_root=root,
        plan=plan,
        plan_ref=plan_ref,
        execution=execution,
        execution_ref=execution_ref,
    )
    return results, write_ref(
        root,
        "artifacts/arena-results.json",
        canonical_json_bytes(results),
    )


def _report_player(reference: ArtifactRef, data: bytes) -> dict[str, object]:
    quantization_code = struct.unpack("<8s10If", data[:52])[9]
    quantization = {0: "float32", 1: "int8"}[quantization_code]
    return {
        "label": f"search:neural:d6:h64:tt-on:book-off:m-{reference.sha256[:12]}",
        "evaluatorKind": "neural",
        "searchDepth": 6,
        "hashMegabytes": 64,
        "transposition": True,
        "modelArtifactSha256": reference.sha256,
        "modelArtifactSize": reference.size,
        "modelPayloadSha256": hashlib.sha256(data[:-32]).hexdigest(),
        "architectureVersion": 1,
        "quantization": quantization,
        "openingEnabled": False,
    }


def make_start_rows(
    *, split: str, prefix: str, base_sfen: str, count: int = 10
) -> list[dict[str, object]]:
    """Create distinct material-valid starts without inventing extra pieces.

    Each fixture position moves one (or, for the last variant, two) white pawns
    from the board into Black's hand.  The former helper added hand pawns without
    removing them from the board, producing impossible 19..28-pawn positions that
    a real Rust SFEN parser correctly rejected.
    """

    rows: list[dict[str, object]] = []
    fields = base_sfen.split(" ")
    ranks = fields[0].split("/")
    if len(ranks) != 9:
        raise ValueError("fixture base SFEN must contain nine board ranks")

    def expand(rank: str) -> list[str]:
        squares: list[str] = []
        for value in rank:
            squares.extend(["."] * int(value) if value.isdigit() else [value])
        if len(squares) != 9:
            raise ValueError("fixture base SFEN rank must contain nine squares")
        return squares

    def compact(squares: list[str]) -> str:
        encoded = ""
        empty = 0
        for value in squares:
            if value == ".":
                empty += 1
                continue
            if empty:
                encoded += str(empty)
                empty = 0
            encoded += value
        return encoded + (str(empty) if empty else "")

    white_pawns = expand(ranks[2])
    if white_pawns != ["p"] * 9:
        raise ValueError("fixture base SFEN must retain all nine white pawns on rank c")
    for index in range(count):
        if index >= 10:
            raise ValueError("material-valid fixture generator supports at most ten rows")
        captured = [index] if index < 9 else [0, 8]
        variant = list(white_pawns)
        for square in captured:
            variant[square] = "."
        variant_ranks = list(ranks)
        variant_ranks[2] = compact(variant)
        hand = "P" if len(captured) == 1 else "2P"
        sfen = " ".join(("/".join(variant_ranks), fields[1], hand, "1"))
        source_game = hashlib.sha256(f"{prefix}-{index}".encode()).hexdigest()
        position_index = index + 1
        rows.append(
            {
                "positionId": start_position_identity(source_game, position_index, sfen),
                "sfen": sfen,
                "sourceGameSha256": source_game,
                "positionIndex": position_index,
                "split": split,
            }
        )
    return rows


_PHASE3_STARTPOS_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
_PHASE3_FIRST_MOVES = (
    "9g9f",
    "8g8f",
    "7g7f",
    "6g6f",
    "5g5f",
    "4g4f",
    "3g3f",
    "2g2f",
    "1g1f",
    "2h1h",
    "2h3h",
    "2h4h",
    "2h5h",
    "2h6h",
    "2h7h",
    "9i9h",
    "7i6h",
    "7i7h",
    "6i5h",
    "6i6h",
)
_PHASE3_CSA_BOARD = (
    "P1-KY-KE-GI-KI-OU-KI-GI-KE-KY\n"
    "P2 * -HI *  *  *  *  * -KA * \n"
    "P3-FU-FU-FU-FU-FU-FU-FU-FU-FU\n"
    "P4 *  *  *  *  *  *  *  *  * \n"
    "P5 *  *  *  *  *  *  *  *  * \n"
    "P6 *  *  *  *  *  *  *  *  * \n"
    "P7+FU+FU+FU+FU+FU+FU+FU+FU+FU\n"
    "P8 * +KA *  *  *  *  * +HI * \n"
    "P9+KY+KE+GI+KI+OU+KI+GI+KE+KY\n"
    "P+\n"
    "P-\n"
    "+\n"
)
_PHASE3_CSA_PIECES = {
    "p": "FU",
    "l": "KY",
    "n": "KE",
    "s": "GI",
    "g": "KI",
    "b": "KA",
    "r": "HI",
    "k": "OU",
}
_PHASE3_CSA_PROMOTED_PIECES = {
    "p": "TO",
    "l": "NY",
    "n": "NK",
    "s": "NG",
    "b": "UM",
    "r": "RY",
}


def _expand_sfen_board(sfen: str) -> list[list[str | None]]:
    ranks = sfen.split(" ", 1)[0].split("/")
    board: list[list[str | None]] = []
    for rank in ranks:
        squares: list[str | None] = []
        promoted = False
        for character in rank:
            if character.isdigit():
                squares.extend([None] * int(character))
            elif character == "+":
                promoted = True
            else:
                squares.append(("+" if promoted else "") + character)
                promoted = False
        if len(squares) != 9 or promoted:
            raise ValueError("fixture SFEN board is malformed")
        board.append(squares)
    if len(board) != 9:
        raise ValueError("fixture SFEN must contain nine ranks")
    return board


def _compact_sfen_board(board: list[list[str | None]]) -> str:
    encoded_ranks: list[str] = []
    for rank in board:
        encoded = ""
        empty = 0
        for piece in rank:
            if piece is None:
                empty += 1
            else:
                if empty:
                    encoded += str(empty)
                    empty = 0
                encoded += piece
        if empty:
            encoded += str(empty)
        encoded_ranks.append(encoded)
    return "/".join(encoded_ranks)


def _apply_fixture_move(
    board: list[list[str | None]], move: str, *, side: str, move_number: int
) -> tuple[str, str]:
    """Apply one preselected non-capturing fixture move and return CSA/SFEN bytes."""

    if len(move) != 4 or side not in {"b", "w"}:
        raise ValueError("fixture move is outside the closed non-promotion subset")
    source_file = int(move[0])
    source_rank = ord(move[1]) - ord("a")
    target_file = int(move[2])
    target_rank = ord(move[3]) - ord("a")
    source_index = 9 - source_file
    target_index = 9 - target_file
    piece = board[source_rank][source_index]
    if piece is None or piece.startswith("+") or board[target_rank][target_index] is not None:
        raise ValueError("fixture move does not move one unpromoted piece to an empty square")
    if (piece.isupper()) != (side == "b"):
        raise ValueError("fixture move side disagrees with the board")
    board[source_rank][source_index] = None
    board[target_rank][target_index] = piece
    marker = "+" if side == "b" else "-"
    csa = (
        f"{marker}{source_file}{source_rank + 1}{target_file}{target_rank + 1}"
        f"{_PHASE3_CSA_PIECES[piece.casefold()]}"
    )
    next_side = "w" if side == "b" else "b"
    sfen = f"{_compact_sfen_board(board)} {next_side} - {move_number + 1}"
    return csa, sfen


def _arena_fixture_csa(
    *,
    sfen: str,
    black: str,
    white: str,
    result: str,
    source_move: str,
) -> tuple[bytes, int, int, int]:
    """Encode one canonical, replayable CSA resignation fixture from an approved start."""

    _board_text, initial_side, hands, move_number = sfen.split()
    if hands != "-" or move_number != "1":
        raise ValueError("arena CSA fixture only supports reset starts without hand pieces")
    if result not in {"black_win", "white_win"}:
        raise ValueError("arena CSA fixture requires a decisive result")
    board = _expand_sfen_board(sfen)
    lines = ["'CSA encoding=UTF-8", "V3.0", f"N+{black}", f"N-{white}"]
    for rank_index, rank in enumerate(board, start=1):
        cells: list[str] = []
        for piece in rank:
            if piece is None:
                cells.append(" * ")
                continue
            promoted = piece.startswith("+")
            letter = piece[-1]
            marker = "+" if letter.isupper() else "-"
            code = (
                _PHASE3_CSA_PROMOTED_PIECES[letter.casefold()]
                if promoted
                else _PHASE3_CSA_PIECES[letter.casefold()]
            )
            cells.append(f"{marker}{code}")
        lines.append(f"P{rank_index}{''.join(cells)}")
    lines.extend(("P+", "P-", "+" if initial_side == "b" else "-"))

    winner = "b" if result == "black_win" else "w"
    moves = 0
    if winner == initial_side:
        csa_move, _ = _apply_fixture_move(
            board,
            source_move,
            side=initial_side,
            move_number=1,
        )
        lines.append(csa_move)
        moves = 1
    elif winner == ("w" if initial_side == "b" else "b"):
        pass
    else:  # pragma: no cover - the two closed result enums make this unreachable
        raise AssertionError("arena CSA fixture winner is outside the two sides")
    lines.append("%TORYO")

    selections = moves + 1
    first = selections // 2 + selections % 2
    second = selections // 2
    black_searches, white_searches = (first, second) if initial_side == "b" else (second, first)
    return ("\n".join(lines) + "\n").encode(), moves, black_searches, white_searches


def _phase3_fixture_source() -> tuple[dict[str, object], tuple[object, ...]]:
    registry = load_source_registry(PROJECT_ROOT / "configs/data_sources.yaml")
    source = registry.get("aobazero-no-noise")
    source_value: dict[str, object] = {
        "sourceId": source.source_id,
        "name": source.name,
        "officialBase": source.official_base,
        "adapter": source.adapter,
        "license": source.license,
        "licenseEvidence": [item.as_dict() for item in source.license_evidence],
        "redistributable": source.redistributable,
        "machineLearningAllowed": source.machine_learning_allowed,
        "lastReviewed": source.last_reviewed.isoformat(),
    }
    return source_value, tuple(sorted(source.catalog, key=lambda item: item.object_id)[:20])


def _phase3_fixture_evidence() -> list[dict[str, object]]:
    snapshots = [
        {
            "evidence_id": "immutable-project-readme-ja",
            "url": (
                "https://raw.githubusercontent.com/kobanium/aobazero/"
                "5eb944165300d5b88924c917a147e80d9d173eed/README.md"
            ),
            "retrieved_at": "2026-08-08T00:00:00Z",
            "sha256": "cfb7bafc4e7942ae8efef2dd9b86bc8d4585c80335400d76b53fa21446220181",
            "size": 5_105,
            "content_type": "text/plain",
            "object_path": (
                "evidence/sha256/cf/"
                "cfb7bafc4e7942ae8efef2dd9b86bc8d4585c80335400d76b53fa21446220181"
            ),
        },
        {
            "evidence_id": "immutable-rights-readme",
            "url": (
                "https://raw.githubusercontent.com/kobanium/aobazero/"
                "5eb944165300d5b88924c917a147e80d9d173eed/README_en.md"
            ),
            "retrieved_at": "2026-08-08T00:00:00Z",
            "sha256": "018bf5496101d0034f8a6559324c298b6adcec88d608e288b6bc49108b6975bc",
            "size": 4_060,
            "content_type": "text/plain",
            "object_path": (
                "evidence/sha256/01/"
                "018bf5496101d0034f8a6559324c298b6adcec88d608e288b6bc49108b6975bc"
            ),
        },
    ]
    return sorted(
        snapshots,
        key=lambda row: (
            row["evidence_id"],
            row["url"],
            row["retrieved_at"],
            row["sha256"],
            row["size"],
            row["content_type"],
            row["object_path"],
        ),
    )


def _phase3_fixture_games() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    source, catalog = _phase3_fixture_source()
    evidence = _phase3_fixture_evidence()
    split_policy = SplitPolicy(
        salt="phase3-test-split-salt",
        test_basis_points=0,
        validation_basis_points=5_000,
    )
    games: list[dict[str, object]] = []
    positions: list[dict[str, object]] = []
    for index, (first_move, catalog_object) in enumerate(
        zip(_PHASE3_FIRST_MOVES, catalog, strict=True)
    ):
        desired_split = "train" if index < 10 else "validation"
        third_move = "7g7f" if first_move == "8g8f" else "8g8f"
        usi_moves = [first_move, "9c9d", third_move]
        board = _expand_sfen_board(_PHASE3_STARTPOS_SFEN)
        sfens = [_PHASE3_STARTPOS_SFEN]
        csa_moves: list[str] = []
        side = "b"
        for move_number, move in enumerate(usi_moves, start=1):
            csa_move, next_sfen = _apply_fixture_move(
                board, move, side=side, move_number=move_number
            )
            csa_moves.append(csa_move)
            sfens.append(next_sfen)
            side = "w" if side == "b" else "b"
        normalized_csa = ""
        canonical_sha256 = ""
        for nonce in range(10_000):
            candidate = (
                "'CSA encoding=UTF-8\nV3.0\n"
                f"$FIXTURE_ID:{index:03}-{nonce:04}\n"
                f"{_PHASE3_CSA_BOARD}" + "\n".join(csa_moves) + "\n%MAX_MOVES\n"
            )
            digest = hashlib.sha256(candidate.encode()).hexdigest()
            if assign_game_split(digest, split_policy) == desired_split:
                normalized_csa = candidate
                canonical_sha256 = digest
                break
        if not normalized_csa:
            raise RuntimeError("fixture nonce search did not produce the requested split")
        raw_lines = [
            f"' fixture acquisition comment {index}",
            f"$FIXTURE_ID:{index:03}-{nonce:04}",
            "PI",
            "+",
        ]
        for move_index, csa_move in enumerate(csa_moves, start=1):
            raw_lines.extend((f"{csa_move},v=0.{move_index},r=0.5", f"T{move_index}"))
        raw_lines.extend(("%MAX_MOVES", "' bounded trailing acquisition comment"))
        raw_csa = "\n".join(raw_lines) + "\n"
        raw_bytes = raw_csa.encode()
        adapted = adapt_aobazero_csa(raw_bytes)
        expected_adapted = (
            "V3.0\n"
            f"$FIXTURE_ID:{index:03}-{nonce:04}\n"
            "PI\n+\n" + "\n".join(csa_moves) + "\n%MAX_MOVES\n"
        )
        if adapted.csa != expected_adapted:
            raise RuntimeError("fixture AobaZero adaptation differs from its staged CSA")
        raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        raw_object = {
            "objectId": catalog_object.object_id,
            "objectPath": f"objects/sha256/{raw_sha256[:2]}/{raw_sha256}",
            "originalFilename": catalog_object.filename,
            "sha256": raw_sha256,
            "size": len(raw_bytes),
            "response": {
                "contentType": "application/x-csa",
                "etag": None,
                "lastModified": None,
            },
        }
        game = {
            "schema": "phase3_game/v1",
            "gameId": canonical_sha256,
            "canonicalSha256": canonical_sha256,
            "rawObject": raw_object,
            "rawCsa": raw_csa,
            "normalizedCsa": normalized_csa,
            "initialSfen": sfens[0],
            "usiMoves": usi_moves,
            "plyCount": len(usi_moves),
            "positionCount": len(sfens),
            "outcome": "unknown",
            "terminalReason": "MAX_MOVES",
            "resultValidation": "external_condition",
            "players": {
                "black": {"name": None, "rating": None},
                "white": {"name": None, "rating": None},
            },
            "date": None,
            "sourceDateTime": None,
            "sourceTimeZone": None,
            "split": desired_split,
            "flags": {"short": True, "long": False},
            "sourceId": source["sourceId"],
            "url": catalog_object.url,
            "retrievedAt": "2026-08-08T00:00:00Z",
            "licenseDecision": {
                "license": source["license"],
                "evidence": source["licenseEvidence"],
                "evidenceSnapshots": evidence,
                "redistributable": source["redistributable"],
                "machineLearningAllowed": source["machineLearningAllowed"],
            },
        }
        games.append(game)
        for position_index, sfen in enumerate(sfens):
            move_usi = usi_moves[position_index] if position_index < len(usi_moves) else None
            next_sfen = sfens[position_index + 1] if position_index + 1 < len(sfens) else None
            terminal_tail = position_index >= 2
            positions.append(
                {
                    "schema": "phase3_position/v1",
                    "gameId": canonical_sha256,
                    "canonicalSha256": canonical_sha256,
                    "rawSha256": raw_sha256,
                    "sourceId": source["sourceId"],
                    "split": desired_split,
                    "positionIndex": position_index,
                    "sfen": sfen,
                    "moveUsi": move_usi,
                    "nextSfen": next_sfen,
                    "outcome": "unknown",
                    "terminalReason": "MAX_MOVES",
                    "sideToMove": "black" if sfen.split(" ")[1] == "b" else "white",
                    "fullPlies": len(usi_moves),
                    "remainingPlies": len(usi_moves) - position_index,
                    "eligible": move_usi is not None and not terminal_tail,
                    "terminalTail": terminal_tail,
                }
            )
    return games, positions


def _write_closed_phase3_fixture(
    root: Path,
) -> tuple[ArtifactRef, ArtifactRef, dict[str, object]]:
    games, positions = _phase3_fixture_games()
    phase3_root = root / "data/phase3"
    games_digest = write_jsonl_gzip_atomic(phase3_root / "games-00000.jsonl.gz", games)
    positions_digest = write_jsonl_gzip_atomic(phase3_root / "positions-00000.jsonl.gz", positions)
    games_ref = ArtifactRef(
        path="data/phase3/games-00000.jsonl.gz",
        sha256=games_digest.sha256,
        size=games_digest.size,
    )
    positions_ref = ArtifactRef(
        path="data/phase3/positions-00000.jsonl.gz",
        sha256=positions_digest.sha256,
        size=positions_digest.size,
    )
    report_ref = write_ref(
        root,
        "data/phase3/normalization-report.json",
        canonical_json_bytes({"schema": "phase3_normalization_report/v1"}),
    )
    source, _ = _phase3_fixture_source()
    split_policy = SplitPolicy(
        salt="phase3-test-split-salt",
        test_basis_points=0,
        validation_basis_points=5_000,
    )
    manifest = {
        "schema": "phase3_dataset_manifest/v1",
        "datasetId": "aobazero-no-noise-contract-fixture",
        "source": source,
        "config": {
            "schema": "phase3_normalization_config/v1",
            "datasetId": "aobazero-no-noise-contract-fixture",
            "exporterTimeoutSeconds": 30,
            "maxGames": len(games),
            "maxPositions": len(positions),
            "maxRawBytes": 1_048_576,
            "split": split_policy.as_dict(),
            "terminalTailPositions": 1,
        },
        "counts": {"games": len(games), "positions": len(positions)},
        "artifacts": {
            "games-00000.jsonl.gz": {
                "sha256": games_ref.sha256,
                "size": games_ref.size,
                "records": len(games),
            },
            "normalization-report.json": {
                "sha256": report_ref.sha256,
                "size": report_ref.size,
                "records": 1,
            },
            "positions-00000.jsonl.gz": {
                "sha256": positions_ref.sha256,
                "size": positions_ref.size,
                "records": len(positions),
            },
        },
        "rawObjectSha256": sorted(str(game["rawObject"]["sha256"]) for game in games),
        "canonicalGameSha256": sorted(str(game["canonicalSha256"]) for game in games),
        "evidenceSnapshots": _phase3_fixture_evidence(),
    }
    return (
        positions_ref,
        write_ref(
            root,
            "data/phase3/manifest.json",
            canonical_json_bytes(manifest),
        ),
        manifest,
    )


def write_validated_phase3_starts(
    root: Path,
    *,
    engine_ref: ArtifactRef,
    train_sfen: str,
    validation_sfen: str,
    git_commit: str = "a399407",
    engine_build_receipt: ArtifactRef | None = None,
) -> tuple[StartPositionSet, ArtifactRef, ArtifactRef, ArtifactRef, ArtifactRef]:
    """Write deterministic Phase 3 source, starts, and immutable Rust-validation evidence."""

    del train_sfen, validation_sfen
    positions_ref, dataset_ref, _ = _write_closed_phase3_fixture(root)
    starts_raw = build_start_positions_from_phase3(
        repository_root=root,
        positions_ref=positions_ref,
        dataset_manifest_ref=dataset_ref,
        seed=20_260_808,
        train_count=10,
        validation_count=10,
    )
    starts_ref = write_ref(
        root,
        "artifacts/phase6-inputs/start-positions.json",
        canonical_json_bytes(starts_raw),
    )
    starts = parse_start_positions(starts_raw)
    validation_results = []
    for position in starts.positions:
        stdout = write_ref(
            root,
            f"artifacts/start-validation/logs/{position.position_id}.stdout.log",
            b"depth 0 nodes 1\n",
        )
        stderr = write_ref(
            root,
            f"artifacts/start-validation/logs/{position.position_id}.stderr.log",
            b"",
        )
        validation_results.append(
            {
                "positionId": position.position_id,
                "sfenSha256": hashlib.sha256(position.sfen.encode()).hexdigest(),
                "legal": True,
                "returnCode": 0,
                "timedOut": False,
                "outputLimitExceeded": False,
                "memoryLimitExceeded": False,
                "peakRssBytes": 1,
                "rssMeasurement": "process_tree_ps_rss_sum",
                "stdout": stdout.as_dict(),
                "stderr": stderr.as_dict(),
                "completedAt": "2026-08-08T00:00:00Z",
            }
        )
    validation = {
        "schema": "phase6_start_position_validation/v1",
        "startPositions": starts_ref.as_dict(),
        "engine": engine_ref.as_dict(),
        "engineBuildReceipt": (
            None if engine_build_receipt is None else engine_build_receipt.as_dict()
        ),
        "gitCommit": git_commit,
        "method": "open-shogi-cli-perft-depth-0",
        "results": validation_results,
    }
    validation_ref = write_ref(
        root,
        "artifacts/start-validation.json",
        canonical_json_bytes(validation),
    )
    return starts, starts_ref, validation_ref, dataset_ref, positions_ref
