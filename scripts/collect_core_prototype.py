#!/usr/bin/env python3
"""Collect one small cohort with the explicit native pure probe. Never opens sealed inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import selectors
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--leaf", type=Path, required=True)
    parser.add_argument("--leaf-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if hashlib.sha256(args.leaf.read_bytes()).hexdigest() != args.leaf_sha256:
        raise ValueError("unexpected leaf model")
    if args.output.exists():
        raise ValueError("output exists; no implicit overwrite or repeated labeling")
    args.output.mkdir(parents=True)
    prefix = [str(args.probe.resolve()), str(args.leaf.resolve()), args.leaf_sha256]
    games, samples = (1, 2) if args.preflight else (24, 8)
    seed = 930_100 if args.preflight else 930_200
    plan = {
        "schema": "open_shogiai_core_collection/v1",
        "preflight": args.preflight,
        "source_games": games,
        "samples_per_game_cap": samples,
        "seed": seed,
        "depth": 4,
        "nodes": 20_000,
        "movetime_ms": 1000,
        "train_games": list(range(16)),
        "validation_games": list(range(16, 20)),
        "development_test_games": list(range(20, 24)),
        "max_updates": 300,
        "retained_checkpoints": 1,
        "retained_resume_states": 1,
        "leaf_sha256": args.leaf_sha256,
        "probe_sha256": hashlib.sha256(args.probe.read_bytes()).hexdigest(),
        "source": (
            "own source-game trajectories, one random move per three, "
            "otherwise W256 depth1/128 nodes"
        ),
        "final_holdout_opened": False,
    }
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    began = time.monotonic()
    generated = subprocess.run(
        [*prefix, "generate", str(games), str(samples), str(seed)],
        capture_output=True,
        text=True,
        check=True,
        timeout=240,
    )
    positions = [json.loads(line) for line in generated.stdout.splitlines()]
    if not positions or len(positions) > games * samples:
        raise ValueError("source generator returned an invalid cohort size")
    process = subprocess.Popen(
        [*prefix, "probe"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=(args.output / "probe.stderr.log").open("w"),
        text=True,
        bufsize=1,
    )
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    depths = []
    elapsed = []
    try:
        with (args.output / "records.jsonl").open("x") as output:
            for index, sample in enumerate(positions):
                request = {
                    "moves": sample["moves"],
                    "depth": plan["depth"],
                    "nodes": plan["nodes"],
                    "movetime_ms": plan["movetime_ms"],
                }
                process.stdin.write(json.dumps(request) + "\n")
                process.stdin.flush()
                if not selector.select(timeout=15):
                    raise TimeoutError("bounded probe stopped responding")
                result = json.loads(process.stdout.readline())
                output.write(json.dumps({"sample": sample, "search": result}) + "\n")
                output.flush()
                depths.append(result["depth"])
                elapsed.append(result["elapsed_ms"])
                print(
                    json.dumps(
                        {
                            "complete": index + 1,
                            "total": len(positions),
                            "depth": result["depth"],
                            "elapsed_ms": result["elapsed_ms"],
                        }
                    ),
                    flush=True,
                )
        process.stdin.close()
        if process.wait(timeout=5) != 0:
            raise RuntimeError("probe failed; keep the incomplete evidence")
    finally:
        selector.close()
        if process.poll() is None:
            process.kill()
            process.wait()
    report = {
        **plan,
        "acquired_positions": len(positions),
        "positions_with_depth_pair": sum(depth >= 2 for depth in depths),
        "depths": depths,
        "wall_seconds": time.monotonic() - began,
        "search_ms_total": sum(elapsed),
        "search_ms_max": max(elapsed),
        "records_sha256": hashlib.sha256((args.output / "records.jsonl").read_bytes()).hexdigest(),
    }
    (args.output / "collection.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
