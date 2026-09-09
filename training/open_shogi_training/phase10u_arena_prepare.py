"""Prepare a review-only immutable schedule from actual audits; never play a game."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import phase10t_pure_build as build
from . import phase10u_arena_evidence as evidence
from . import phase10u_prior_audit as adapter
from .phase10u_arena_runner import Player


def reference(root: Path, path: Path) -> dict:
    return {"path": str(path.relative_to(root)), "sha256": evidence.file_sha(path)}


def publish(root: Path, source: Path, name: str) -> dict:
    """Exclusive content-addressed executable publication, retaining all original builds."""
    sha = evidence.file_sha(source)
    folder = root / "local/phase10u-certified" / sha
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    if not path.exists():
        with path.open("xb") as handle:
            handle.write(source.read_bytes())
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(0o555)
    if evidence.file_sha(path) != sha:
        raise evidence.EvidenceError("published binary differs from certified build")
    return reference(root, path)


def prepare(root: Path, audit_path: Path, output: Path) -> dict:
    certified = json.loads(audit_path.read_text())
    adapter.verify_evidence(root, certified)
    audit = build.Audit(root, output)
    commit = audit.run(["git", "rev-parse", "HEAD"]).strip()
    if audit.run(["git", "status", "--porcelain"]) or commit != certified["git_commit"]:
        raise evidence.EvidenceError("preparation requires the audited clean source commit")
    audit.run(
        [
            "cargo",
            "build",
            "--locked",
            "--release",
            "-p",
            "open-shogi-cli",
            "--no-default-features",
            "--features",
            "handcrafted",
        ]
    )
    audit.run(
        [
            "cargo",
            "tree",
            "--locked",
            "-p",
            "open-shogi-cli",
            "-e",
            "features",
            "--no-default-features",
            "--features",
            "handcrafted",
        ]
    )
    audit.run(
        [
            "cargo",
            "build",
            "--locked",
            "--release",
            "-p",
            "open-shogi-core",
            "--example",
            "phase10u_replay",
            "--no-default-features",
            "--features",
            "pure-only",
        ]
    )
    pure_native = publish(root, root / certified["artifacts"]["native"]["path"], "open-shogi-cli")
    handcrafted_native = publish(root, output / "target/release/open-shogi-cli", "open-shogi-cli")
    oracle = publish(root, output / "target/release/examples/phase10u_replay", "phase10u-replay")
    players = {}
    for model in certified["models"]:
        identity = model["identity"]
        profile_path = root / (
            "configs/runtime/pure_learned-v1.json"
            if identity["format"] == "OSAVAL02"
            else "configs/runtime/pure_learned-a1-v1.json"
        )
        players[identity["id"]] = {
            "source_commit": commit,
            "cargo_features": ["pure-only"],
            "adapter_format": identity["format"],
            "profile_name": "pure_learned",
            "evaluator_schema_sha256": evidence.file_sha(profile_path),
            "native": pure_native,
            "wasm": certified["artifacts"]["wasm"],
            "profile": reference(root, profile_path),
            "model": {
                "path": identity["path"],
                "sha256": identity["sha256"],
                "payload_sha256": identity["payload_sha256"],
            },
        }
    profile_path = root / "configs/runtime/handcrafted-experimental-v1.json"
    players["handcrafted"] = {
        "source_commit": commit,
        "cargo_features": ["handcrafted"],
        "adapter_format": "HANDCRAFTED",
        "profile_name": "handcrafted_experimental",
        "evaluator_schema_sha256": evidence.file_sha(profile_path),
        "native": handcrafted_native,
        "wasm": None,
        "model": None,
        "profile": reference(root, profile_path),
    }
    # These are transport/identity probes on a synthetic position, never scored games.
    initial = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
    for name, identity in players.items():
        player = Player(root, identity, output / f"{name}-transport.log")
        try:
            request = {
                "initial_sfen": initial,
                "moves": [],
                "depth": 1,
                "nodes": 4,
                "movetime_ms": None,
                "hard_timeout_ms": 30000,
                "hash_mb": 32,
            }
            response, timing = player.search(request)
            counters = {
                key: response["proof"][key] for key in (*evidence.PROHIBITED, "learned_eval_calls")
            }
            evidence.proof(counters, handcrafted=name == "handcrafted")
            if response["timing"]["hard_compliant"] is not True or not timing["compliant"]:
                raise evidence.EvidenceError("transport probe exceeded hard deadline")
        finally:
            player.close()
        receipt = {
            **identity,
            "status": "PASS",
            "prohibited_modules_absent": name != "handcrafted",
            "parent_adapter_audit": reference(root, audit_path) if name != "handcrafted" else None,
            "source_hashes": build.source_inventory(root),
            "build_commands": audit.commands,
            "transport_probe": {
                "response": response,
                "timing": timing,
                "log": reference(root, output / f"{name}-transport.log"),
            },
        }
        path = output / f"{name}-build-audit.json"
        path.write_bytes(evidence.canonical(receipt))
        identity["build_audit"] = reference(root, path)
    plan = json.loads((root / "configs/phase10u/arena-rerun-plan.json").read_text())
    expected_controls = {
        "threads_per_player": 1,
        "hash_mib_per_player": 32,
        "depth_cap": 64,
        "max_plies": 256,
        "book": False,
        "nodes": 2000,
        "node_emergency_timeout_ms": 30000,
        "clock_ms": 100,
        "clock_hard_cap_ms": 1000,
    }
    if (
        evidence.canonical(plan["controls_values"]) != evidence.canonical(expected_controls)
        or plan["modes"] != ["equal_nodes", "equal_wall_clock"]
        or len(plan["starts"]) != 100
        or plan["total_new_games"] != 1200
        or plan["execution_authorized_in_this_task"] is not False
    ):
        raise evidence.EvidenceError("frozen schedule/control drift")
    starts_path = output / "starts.json"
    starts_path.write_bytes(evidence.canonical(plan["starts"]))
    games = {}
    for entrant in plan["entrants"]:
        for mode in plan["modes"]:
            clock = mode == "equal_wall_clock"
            controls = {
                "clock": "monotonic_ns",
                "threads": 1,
                "hash_mb": 32,
                "max_depth": 64,
                "nodes": None if clock else 2000,
                "movetime_ns": 100_000_000 if clock else None,
                "hard_timeout_ns": 1_000_000_000 if clock else 30_000_000_000,
                "max_plies": 256,
                "search_options": {
                    "configuration": "SearchConfig::default",
                    "hash_mb": 32,
                    "fresh_engine_per_move": True,
                    "book": False,
                },
                "warmup": "none; model startup excluded for both sides; all request setup charged",
            }
            for index, start in enumerate(plan["starts"]):
                pairing = f"{entrant['id']}-{mode}-{index:03}"
                for reverse in (False, True):
                    sides = [players[entrant["id"]], players["handcrafted"]]
                    if reverse:
                        sides.reverse()
                    games[f"{pairing}-{int(reverse)}"] = {
                        "initial_sfen": start["sfen"],
                        "seed": plan["seed"],
                        "pairing_id": pairing,
                        "controls": controls,
                        "sides": dict(zip(("black", "white"), sides, strict=True)),
                    }
    manifest = {
        "schema": "open_shogiai_phase10u_reviewed_arena_manifest/v1",
        "status": "READY_FOR_REVIEW_NOT_EXECUTED",
        "execution_authorized": False,
        "source_commit": commit,
        "original_start_manifest": plan["start_manifest"],
        "start_manifest": reference(root, starts_path),
        "replay_oracle": oracle,
        "games": games,
        "process_parallelism": 1,
        "planned_games": len(games),
    }
    evidence.validate_manifest(root, manifest, evidence.digest(manifest))
    path = output / "reviewed-manifest.json"
    path.write_bytes(evidence.canonical(manifest))
    return reference(root, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    print(json.dumps(prepare(root, args.adapter_audit.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
