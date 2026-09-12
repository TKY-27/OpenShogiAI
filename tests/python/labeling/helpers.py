from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from open_shogi_training.data.gzip_jsonl import write_jsonl_gzip_atomic
from open_shogi_training.labeling.artifacts import compact_json_bytes, write_json_atomic
from open_shogi_training.labeling.benchmark import BENCHMARK_SCHEMA
from open_shogi_training.labeling.config import TeacherConfig, load_teacher_config
from open_shogi_training.labeling.fingerprint import fingerprint_teacher
from open_shogi_training.labeling.selection import SelectionResult, select_positions

FAKE_ENGINE = r"""#!{python}
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="good")
parser.add_argument("--marker")
parser.add_argument("--child-pid")
parser.add_argument("--command-log")
parser.add_argument("--fail-token", default="FAIL")
args = parser.parse_args()
position = ""

def emit(line):
    print(line, flush=True)

def log(line):
    if args.command_log:
        with Path(args.command_log).open("a", encoding="utf-8") as output:
            output.write(line + "\n")

def good_search():
    if args.mode == "stderr":
        sys.stderr.write("S" * 10000 + "\n")
        sys.stderr.flush()
    emit("info depth 8 seldepth 12 nodes 20 multipv 1 score cp 42 pv 7g7f 3c3d")
    emit("info depth 8 seldepth 11 nodes 19 multipv 2 score mate -3 pv 2g2f 8c8d")
    emit("info depth 8 seldepth 10 nodes 18 multipv 3 score cp -7 pv 5g5f 5c5d")
    emit("bestmove 7g7f ponder 3c3d")

for raw in sys.stdin:
    line = raw.rstrip("\r\n")
    log(line)
    if line == "usi":
        identity_name = "Fake Teacher 1.0"
        if args.mode == "crash-identity-drift" and args.marker and Path(args.marker).exists():
            identity_name = "Fake Teacher drifted"
        emit(f"id name {{identity_name}}")
        emit("id author Test Suite")
        for name, kind in [
            ("Book_Enable", "check"),
            ("Clear_Hash", "button"),
            ("Eval_Dir", "string"),
            ("Eval_Hash", "spin"),
            ("MultiPV", "spin"),
            ("Threads", "spin"),
            ("USI_Hash", "spin"),
            ("USI_Ponder", "check"),
        ]:
            emit(f"option name {{name}} type {{kind}} default 1")
        emit("usiok")
    elif line == "isready":
        emit("readyok")
    elif line.startswith("position sfen "):
        position = line.removeprefix("position sfen ")
    elif line.startswith("go nodes "):
        if (
            args.mode in {{"crash-once", "crash-identity-drift"}}
            and args.marker
            and not Path(args.marker).exists()
        ):
            Path(args.marker).write_text("crashed", encoding="utf-8")
            sys.stderr.write("intentional first crash\n")
            sys.stderr.flush()
            os._exit(7)
        if args.mode == "malformed" or (args.mode == "fail-token" and args.fail_token in position):
            emit("info depth 8 seldepth 12 nodes 20 multipv 1 score bananas 4 pv 7g7f")
            emit("bestmove 7g7f")
            continue
        if args.mode in {{"bounded-then-good", "bounded-only"}}:
            emit(
                "info depth 7 seldepth 10 nodes 10 multipv 1 score cp 50 "
                "lowerbound nps 100 time 1 pv 7g7f"
            )
            emit(
                "info depth 7 seldepth 10 nodes 10 multipv 2 score mate -2 "
                "upperbound nps 100 time 1 pv 2g2f"
            )
            emit(
                "info depth 7 seldepth 10 nodes 10 multipv 3 score cp -9 "
                "upperbound nps 100 time 1 pv 5g5f"
            )
            if args.mode == "bounded-only":
                emit("bestmove 7g7f")
                continue
        if args.mode == "later-lower-exact":
            emit("info depth 8 seldepth 12 nodes 16 multipv 1 score cp 60 pv 7g7f")
            emit("info depth 8 seldepth 12 nodes 16 multipv 2 score cp 50 pv 2g2f")
            emit("info depth 8 seldepth 12 nodes 16 multipv 3 score cp 40 pv 5g5f")
            emit("info depth 7 seldepth 11 nodes 20 multipv 1 score cp 55 pv 2g2f")
            emit("info depth 6 seldepth 10 nodes 20 multipv 2 score cp 45 pv 7g7f")
            emit("info depth 6 seldepth 10 nodes 20 multipv 3 score cp 35 pv 5g5f")
            emit("bestmove 2g2f")
            continue
        if args.mode == "illegal-second-pv":
            emit("info depth 8 seldepth 12 nodes 20 multipv 1 score cp 42 pv 7g7f 3c3d")
            emit("info depth 8 seldepth 11 nodes 19 multipv 2 score cp 7 pv 7g7g")
            emit("info depth 8 seldepth 10 nodes 18 multipv 3 score cp -7 pv 5g5f 5c5d")
            emit("bestmove 7g7f")
            continue
        if args.mode in {{"one-rank", "two-ranks", "gapped-ranks"}}:
            emit("info depth 8 seldepth 12 nodes 20 multipv 1 score cp 42 pv 7g7f")
            if args.mode == "two-ranks":
                emit("info depth 8 seldepth 11 nodes 19 multipv 2 score cp 7 pv 2g2f")
            if args.mode == "gapped-ranks":
                emit("info depth 8 seldepth 10 nodes 18 multipv 3 score cp -7 pv 5g5f")
            emit("bestmove 7g7f")
            continue
        if args.mode == "overflow":
            emit("x" * 10000)
            continue
        if args.mode == "timeout-child":
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            if args.child_pid:
                Path(args.child_pid).write_text(str(child.pid), encoding="utf-8")
            time.sleep(60)
            continue
        good_search()
    elif line == "stop":
        emit("bestmove 7g7f")
    elif line == "quit":
        raise SystemExit(0)
"""

FAKE_OPEN_SHOGI_CLI = r"""#!{python}
import sys

if sys.argv[1:] == ["--version"]:
    print("OpenShogiAI 0.0.0")
    raise SystemExit(0)
if sys.argv[1:2] == ["perft"]:
    sfen = sys.argv[sys.argv.index("--sfen") + 1]
    moves = ["7g7f", "2g2f", "5g5f"]
    if "forced-one" in sfen:
        moves = moves[:1]
    elif "forced-two" in sfen:
        moves = moves[:2]
    for move in moves:
        print(f"{{move}} 1")
    print(
        f"depth 1 nodes {{len(moves)}} captures 0 promotions 0 drops 0 checks 0 checkmates 0"
    )
    raise SystemExit(0)
if sys.argv[1:] != ["usi"]:
    raise SystemExit(2)
for raw in sys.stdin:
    line = raw.rstrip("\r\n")
    if line == "usi":
        print("id name OpenShogiAI 0.0.0", flush=True)
        print("id author OpenShogiAI contributors", flush=True)
        print("usiok", flush=True)
    elif line.startswith("position sfen ") and " 7g7g" in line:
        print("info string error move 1 is illegal: test rejection", flush=True)
    elif line == "isready":
        print("readyok", flush=True)
    elif line == "quit":
        raise SystemExit(0)
"""


def make_fake_project(
    root: Path,
    *,
    mode: str = "good",
    extra_arguments: list[str] | None = None,
    max_positions: int = 10,
    max_quarantined: int = 5,
    max_retries: int = 1,
    search_ms: int = 500,
    max_stdout_line_bytes: int = 4096,
    nodes: int = 20,
    max_input_rows: int = 1_000,
    max_input_compressed_bytes: int = 1024 * 1024,
    max_input_uncompressed_bytes: int = 4 * 1024 * 1024,
) -> tuple[TeacherConfig, Path, Path]:
    engine = root / "bin" / "fake_teacher.py"
    engine.parent.mkdir(parents=True)
    engine.write_text(FAKE_ENGINE.format(python=sys.executable), encoding="utf-8")
    engine.chmod(0o755)
    engine_sha256 = hashlib.sha256(engine.read_bytes()).hexdigest()
    validator = root / "target" / "release" / "open-shogi-cli"
    validator.parent.mkdir(parents=True)
    validator.write_text(FAKE_OPEN_SHOGI_CLI.format(python=sys.executable), encoding="utf-8")
    validator.chmod(0o755)
    eval_dir = root / "eval"
    eval_dir.mkdir()
    eval_files = []
    for name, data in (("KKP.bin", b"small-kkp"), ("KPP.bin", b"small-kpp")):
        path = eval_dir / name
        path.write_bytes(data)
        eval_files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    arguments = ["--mode", mode, *(extra_arguments or [])]
    payload = {
        "schema": "phase4_teacher_config/v1",
        "teacher": {
            "name": "Fake Teacher",
            "version": "1.0",
            "executable": engine.relative_to(root).as_posix(),
            "arguments": arguments,
            "cwd": engine.parent.relative_to(root).as_posix(),
            "install_manifest": None,
            "binary_sha256": engine_sha256,
            "eval_dir": eval_dir.relative_to(root).as_posix(),
            "eval_files": eval_files,
            "options": {
                "Book_Enable": False,
                "Eval_Dir": "../eval",
                "Eval_Hash": 16,
                "USI_Ponder": False,
            },
            "nodes": nodes,
            "multipv": 3,
            "threads": 1,
            "hash_mb": 16,
            "concurrency": 1,
            "reference_host_memory_gib": 24,
            "working_memory_limit_gib": 20,
        },
        "timeouts": {
            "startup_ms": 5000,
            "ready_ms": 5000,
            "search_ms": search_ms,
            "stop_ms": 100,
            "quit_ms": 100,
        },
        "protocol_limits": {
            "max_stdout_line_bytes": max_stdout_line_bytes,
            "max_stdout_queue_lines": 128,
            "max_search_lines": 1000,
            "max_stderr_bytes": 512,
        },
        "selection": {
            "seed": "test-selection-seed",
            "max_positions": max_positions,
            "max_positions_per_game": max_positions,
            "opening_end_basis_points": 3333,
            "middlegame_end_basis_points": 6667,
            "max_input_rows": max_input_rows,
            "max_input_compressed_bytes": max_input_compressed_bytes,
            "max_input_uncompressed_bytes": max_input_uncompressed_bytes,
            "max_input_line_bytes": 64 * 1024,
        },
        "benchmark": {
            "node_candidates": [10, nodes] if nodes != 10 else [nodes],
            "positions": 1,
            "max_p95_ms": 10000,
            "max_peak_rss_mib": 1024,
            "require_peak_rss": False,
            "rss_poll_ms": 10,
        },
        "labeling": {
            "max_retries": max_retries,
            "max_quarantined": max_quarantined,
            "manifest_interval": 2,
        },
    }
    config_path = root / "teacher.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return load_teacher_config(config_path), engine, config_path


def phase3_position(
    *,
    game: str,
    index: int,
    sfen: str,
    split: str = "train",
    full_plies: int = 30,
    eligible: bool = True,
    outcome: str = "black_win",
) -> dict[str, Any]:
    game_id = hashlib.sha256(game.encode()).hexdigest()
    side = "black" if sfen.split(" ")[1] == "b" else "white"
    return {
        "schema": "phase3_position/v1",
        "gameId": game_id,
        "canonicalSha256": game_id,
        "rawSha256": hashlib.sha256(f"raw-{game}".encode()).hexdigest(),
        "sourceId": "test-source",
        "split": split,
        "positionIndex": index,
        "sfen": sfen,
        "moveUsi": "7g7f" if eligible else None,
        "nextSfen": f"next-{game}-{index} w - {index + 2}" if eligible else None,
        "outcome": outcome,
        "terminalReason": "TORYO",
        "sideToMove": side,
        "fullPlies": full_plies,
        "remainingPlies": full_plies - index,
        "eligible": eligible,
        "terminalTail": False,
    }


def write_phase3_dataset(root: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    positions = root / "positions-00000.jsonl.gz"
    digest = write_jsonl_gzip_atomic(positions, rows)
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "phase3_dataset_manifest/v1",
                "artifacts": {
                    positions.name: {
                        "sha256": digest.sha256,
                        "size": digest.size,
                        "records": digest.records,
                    }
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return positions, manifest


def write_authorized_benchmark(
    root: Path,
    config: TeacherConfig,
    selection: SelectionResult,
) -> Path:
    fingerprint = fingerprint_teacher(config, root)
    teacher_identity_sha256 = hashlib.sha256(
        compact_json_bytes(fingerprint.identity_record())
    ).hexdigest()
    report = {
        "schema": BENCHMARK_SCHEMA,
        "created_at": "2026-08-08T00:00:00.000Z",
        "config_sha256": config.sha256,
        "dataset_manifest_sha256": selection.dataset_manifest_sha256,
        "positions_sha256": selection.positions_sha256,
        "selection_sha256": selection.selection_sha256,
        "sample_position_ids": [selection.positions[0].position_id],
        "teacher": fingerprint.identity_record(),
        "teacher_identity_sha256": teacher_identity_sha256,
        "reported_identity": {"name": "Fake Teacher 1.0", "author": "Test Suite"},
        "resources": {
            "reference_host_memory_gib": config.reference_host_memory_gib,
            "working_memory_limit_gib": config.working_memory_limit_gib,
            "concurrency": config.concurrency,
            "threads": config.threads,
            "usi_hash_mb": config.hash_mb,
            "rss_measurement": "ps process RSS polled during each search",
            "rss_poll_ms": config.benchmark.rss_poll_ms,
            "max_peak_rss_mib": config.benchmark.max_peak_rss_mib,
            "require_peak_rss": config.benchmark.require_peak_rss,
            "max_p95_ms": config.benchmark.max_p95_ms,
        },
        "budgets": [
            {
                "nodes": candidate,
                "searches": [
                    {
                        "position_id": selection.positions[0].position_id,
                        "canonical_state_sha256": selection.positions[0].canonical_state_sha256,
                        "teacher_identity_sha256": teacher_identity_sha256,
                        "status": "completed",
                        "error_category": None,
                        "error_message": None,
                        "elapsed_ms": 1,
                        "reported_nodes": candidate,
                        "depth": 1,
                        "peak_rss_bytes": 1024,
                    }
                ],
                "completed": 1,
                "failed": 0,
                "elapsed_ms": {"min": 1, "median": 1, "p95": 1, "max": 1},
                "peak_rss_bytes": 1024,
                "memory_measured_for_all": True,
                "passes": True,
            }
            for candidate in config.benchmark.node_candidates
        ],
        "selected_nodes": config.nodes,
        "configured_nodes": config.nodes,
        "labeling_authorized": True,
    }
    path = root / "benchmark.json"
    write_json_atomic(path, report, replace=False)
    return path


def selection_for(
    config: TeacherConfig,
    positions: Path,
    manifest: Path,
) -> SelectionResult:
    return select_positions(positions, manifest, config.selection)
