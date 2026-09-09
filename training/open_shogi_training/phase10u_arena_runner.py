"""Separate immutable player coordinator. Execution requires a later pinned authorization.

The review manifest never authorizes execution. This transport retains failed attempts,
checks each launch and every response, and delegates adjudication to the rules-only oracle.
"""

from __future__ import annotations

import argparse
import json
import os
import selectors
import subprocess
import time
from pathlib import Path

from . import phase10u_arena_evidence as evidence


class Player:
    """Bounded JSON-lines transport; no mutable target/release executable is accepted."""

    def __init__(self, root: Path, identity: dict, log: Path):
        self.root, self.identity = root, identity
        native = identity["native"]
        evidence.artifact(root, native)
        if native["sha256"] not in Path(native["path"]).parts:
            raise evidence.EvidenceError("player executable must be content-addressed")
        argv = [str(root / native["path"]), "arena-player", "--profile", identity["profile_name"]]
        if identity["model"] is not None:
            evidence.artifact(root, identity["model"])
            argv += [
                "--model",
                str(root / identity["model"]["path"]),
                "--model-sha256",
                identity["model"]["sha256"],
                "--model-format",
                identity["adapter_format"],
            ]
        self.log = log.open("xb")
        self.pending = b""
        try:
            self.process = subprocess.Popen(
                argv,
                cwd=root,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                bufsize=0,
            )
        except BaseException:
            self.log.close()
            raise
        os.set_blocking(self.process.stdin.fileno(), False)
        try:
            ready = self.receive(time.monotonic_ns() + 30_000_000_000)
            expected_hash = identity["model"]["sha256"] if identity["model"] else None
            if (
                ready["schema"] != "open_shogiai_phase10u_arena_player_ready/v1"
                or ready["profile"] != identity["profile_name"]
                or ready["adapter_format"] != identity["adapter_format"]
                or ready["model_sha256"] != expected_hash
            ):
                raise evidence.EvidenceError("player startup identity mismatch")
            evidence.artifact(root, native)
            self.ready = ready
        except BaseException:
            self.close()
            raise

    def receive(self, deadline_ns: int) -> dict:
        if time.monotonic_ns() >= deadline_ns:
            raise evidence.EvidenceError("player response deadline exceeded")
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while b"\n" not in self.pending:
                remaining = (deadline_ns - time.monotonic_ns()) / 1_000_000_000
                if remaining <= 0 or not selector.select(remaining):
                    raise evidence.EvidenceError("player response deadline exceeded")
                chunk = os.read(self.process.stdout.fileno(), 65536)
                if not chunk:
                    raise evidence.EvidenceError("player exited before complete response")
                self.pending += chunk
                if len(self.pending) > 1_000_000:
                    raise evidence.EvidenceError("oversized player response")
            line, self.pending = self.pending.split(b"\n", 1)
            self.log.write(line + b"\n")
            self.log.flush()
            value = json.loads(line)
            if not isinstance(value, dict):
                raise evidence.EvidenceError("player response is not an object")
            return value

    def send(self, payload: bytes, deadline_ns: int) -> None:
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdin, selectors.EVENT_WRITE)
            while payload:
                remaining = (deadline_ns - time.monotonic_ns()) / 1_000_000_000
                if remaining <= 0 or not selector.select(remaining):
                    raise evidence.EvidenceError("player request deadline exceeded")
                try:
                    count = os.write(self.process.stdin.fileno(), payload)
                except BlockingIOError:
                    continue
                payload = payload[count:]

    def search(self, request: dict) -> tuple[dict, dict]:
        evidence.artifact(self.root, self.identity["native"])
        timer = evidence.MonotonicSearch(request["hard_timeout_ms"] * 1_000_000)
        payload = evidence.canonical(request)
        if len(payload) > 16384:
            raise evidence.EvidenceError("oversized player request")
        self.send(payload, timer.deadline_ns)
        response = self.receive(timer.deadline_ns)
        timing = timer.finish()
        if not timing["compliant"]:
            raise evidence.EvidenceError("per-search coordinator deadline exceeded")
        if self.pending:
            raise evidence.EvidenceError("unsolicited player output")
        if evidence.canonical(response.get("requested_controls")) != evidence.canonical(request):
            raise evidence.EvidenceError("player controls mismatch")
        if response.get("model_format") != self.identity["adapter_format"]:
            raise evidence.EvidenceError("player format mismatch")
        expected_hash = self.identity["model"]["sha256"] if self.identity["model"] else ""
        if response.get("model_sha256") != expected_hash:
            raise evidence.EvidenceError("player model mismatch")
        evidence.artifact(self.root, self.identity["native"])
        return response, timing

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        for stream in (self.process.stdin, self.process.stdout):
            if stream:
                stream.close()
        self.log.close()


def replay(root: Path, trusted: dict, initial: str, moves: list[str], cap: int) -> dict:
    ref = trusted["replay_oracle"]
    evidence.artifact(root, ref)
    result = subprocess.run(
        [str(root / ref["path"])],
        input=evidence.canonical({"initial_sfen": initial, "moves": moves, "max_plies": cap}),
        capture_output=True,
        timeout=30,
        check=True,
    )
    evidence.artifact(root, ref)
    return json.loads(result.stdout)


def require_authorization(trusted: dict, trusted_sha256: str, authorization: dict) -> None:
    if evidence.digest(trusted) != trusted_sha256:
        raise evidence.EvidenceError("reviewed manifest digest mismatch")
    if evidence.canonical(authorization) != evidence.canonical(
        {
            "schema": "open_shogiai_phase10u_later_arena_authorization/v1",
            "manifest_sha256": trusted_sha256,
            "execution_authorized": True,
        }
    ):
        raise evidence.EvidenceError("separate later execution authorization is required")


def run(root: Path, trusted: dict, trusted_sha256: str, authorization: dict, output: Path) -> None:
    """Execute only an explicitly authorized exact schedule, never resume by overwriting."""
    require_authorization(trusted, trusted_sha256, authorization)
    evidence.validate_manifest(root, trusted, trusted_sha256)
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_bytes(evidence.canonical(trusted))
    (output / "authorization.json").write_bytes(evidence.canonical(authorization))
    try:
        for game_id, game in trusted["games"].items():
            _game(root, trusted, trusted_sha256, game_id, game, output)
    except BaseException as error:
        (output / "STOP_CLOSED.json").write_bytes(
            evidence.canonical(
                {
                    "status": "STOP_CLOSED",
                    "error": str(error),
                    "games_not_resumed": True,
                }
            )
        )
        raise


def _game(
    root: Path, trusted: dict, manifest_hash: str, game_id: str, game: dict, output: Path
) -> None:
    if not game_id or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in game_id
    ):
        raise evidence.EvidenceError("unsafe game id")
    players = {}
    try:
        for side in ("black", "white"):
            players[side] = Player(root, game["sides"][side], output / f"{game_id}-{side}.log")
        moves, searches = [], []
        controls = game["controls"]
        current = replay(root, trusted, game["initial_sfen"], moves, controls["max_plies"])
        while current["termination"] == "ongoing":
            side = current["side_to_move"]
            request = {
                "initial_sfen": game["initial_sfen"],
                "moves": moves,
                "depth": controls["max_depth"],
                "nodes": controls["nodes"],
                "movetime_ms": controls["movetime_ns"] // 1_000_000
                if controls["movetime_ns"]
                else None,
                "hard_timeout_ms": controls["hard_timeout_ns"] // 1_000_000,
                "hash_mb": controls["hash_mb"],
            }
            response, timing = players[side].search(request)
            movement = response["best_move"]
            if not isinstance(movement, str):
                raise evidence.EvidenceError("missing legal move; no scored resignation fallback")
            if response["final_sfen"] != current["final_sfen"]:
                raise evidence.EvidenceError("player searched wrong replay position")
            counters = {
                key: response["proof"][key] for key in (*evidence.PROHIBITED, "learned_eval_calls")
            }
            searches.append(
                {
                    "side": side,
                    "move": movement,
                    "timing": timing,
                    "engine_timing": response["timing"],
                    "runtime": counters,
                    "response": response,
                    "request": json.loads(evidence.canonical(request)),
                }
            )
            moves.append(movement)
            current = replay(root, trusted, game["initial_sfen"], moves, controls["max_plies"])
        receipt = {
            "schema": evidence.SCHEMA,
            "manifest_sha256": manifest_hash,
            "game_id": game_id,
            **game,
            "start_manifest_sha256": trusted["start_manifest"]["sha256"],
            "moves": moves,
            "searches": searches,
            "runtime": {
                side: {
                    key: sum(s["runtime"][key] for s in searches if s["side"] == side)
                    for key in (*evidence.PROHIBITED, "learned_eval_calls")
                }
                for side in ("black", "white")
            },
            **{
                key: current[key]
                for key in ("final_sfen", "termination", "result", "max_plies_reached")
            },
        }
        receipt = evidence.seal(receipt)
        evidence.validate(root, receipt, trusted, manifest_hash)
        evidence.write_immutable(output / f"{game_id}.json", receipt)
    finally:
        for player in players.values():
            player.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.root.resolve(),
        json.loads(args.manifest.read_text()),
        args.manifest_sha256,
        json.loads(args.authorization.read_text()),
        args.output,
    )


if __name__ == "__main__":
    main()
