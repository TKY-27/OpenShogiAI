"""Two explicitly started local clock games with a shared frozen W256 leaf.

This is a small prototype comparison, not a frozen campaign or a strength certificate.
All invocations are read-only toward existing source/models; output stays under local/.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import resource
import selectors
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "local/core-prototype/comparison"
LEAF_SHA = "859e922b3f503ddeecf0afeb9a05fccac080a9faca3b19fce9d8253c9039c480"
CONTROL_SHA = "66110bae4ef5fbedd6a3c5537813f0fae5b9276b576a745b2364b4a093141b63"
STARTPOS = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
CLOCK_NS = 20_000_000_000
MAX_PLIES = 160
DEPTH = 5
RESPONSE_LIMIT = 8_000_000
FORBIDDEN = (
    "handcrafted_eval_calls",
    "residual_eval_calls",
    "composite_eval_calls",
    "book_hits",
    "teacher_calls",
    "fallback_count",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON value: {value}")


def regular_inside(path: Path) -> Path:
    path = path if path.is_absolute() else ROOT / path
    path = path.resolve(strict=True)
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f"not a regular repository file: {path}")
    return path


def artifact(path: Path, expected: str | None = None) -> dict[str, Any]:
    requested = path if path.is_absolute() else ROOT / path
    path = regular_inside(path)
    digest = sha(path)
    if expected is not None and digest != expected:
        raise ValueError(f"artifact SHA-256 mismatch: {path.relative_to(ROOT)}")
    return {
        "path": str(path.relative_to(ROOT)),
        "requested_path": str(requested.relative_to(ROOT)),
        "sha256": digest,
        "bytes": path.stat().st_size,
    }


def snapshot(ref: dict[str, Any], folder: Path, name: str) -> dict[str, Any]:
    source, destination = ROOT / ref["path"], folder / name
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst)
    destination.chmod(source.stat().st_mode & 0o777)
    return artifact(destination, ref["sha256"])


def cpu_usage() -> dict[str, float]:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {"user_seconds": usage.ru_utime, "system_seconds": usage.ru_stime}


def cpu_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {key: after[key] - before[key] for key in before}


class TraceJournal:
    """Keep one compressed trace and bind it to the exact uncompressed event bytes."""

    def __init__(self, path: Path):
        self.path = path
        self.raw = path.open("xb")
        self.stream = gzip.GzipFile(filename="", mode="wb", fileobj=self.raw, mtime=0)
        self.digest = hashlib.sha256()
        self.byte_count = 0

    def write(self, data: bytes) -> None:
        self.stream.write(data)
        self.digest.update(data)
        self.byte_count += len(data)

    def flush(self) -> None:
        self.stream.flush()

    def close(self) -> dict[str, Any]:
        self.stream.close()
        self.raw.close()
        restored = hashlib.sha256()
        restored_bytes = 0
        with gzip.open(self.path, "rb") as stream:
            while block := stream.read(1_048_576):
                restored.update(block)
                restored_bytes += len(block)
        if restored.hexdigest() != self.digest.hexdigest() or restored_bytes != self.byte_count:
            raise ValueError("compressed trace did not restore the exact original bytes")
        return {
            **artifact(self.path),
            "encoding": "gzip JSONL",
            "uncompressed_sha256": self.digest.hexdigest(),
            "uncompressed_bytes": self.byte_count,
            "decompression_verified": True,
            "game_binding": "each event has game_id matching the compact receipt",
        }


def remove_duplicate(copy: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """Delete only an unused verified local comparison copy with a retained source."""
    path = ROOT / copy["path"]
    result = {**copy, "reason": "duplicate of retained verified controller", "removed": False}
    if path.is_symlink() or not path.resolve().is_relative_to(OUTPUT_ROOT):
        raise ValueError("duplicate cleanup path escaped the comparison directory")
    artifact(path, source["sha256"])
    artifact(Path(source["path"]), copy["sha256"])
    if not shutil.which("lsof"):
        return {**result, "retained_reason": "open-file check unavailable"}
    opened = subprocess.run(["lsof", "--", str(path)], capture_output=True, text=True, check=False)
    if opened.returncode != 1 or opened.stdout or opened.stderr:
        return {**result, "retained_reason": "open-file check did not confirm unused"}
    path.unlink()
    return {**result, "removed": True, "open_file_check": "lsof: no open handles"}


class ProbeError(Exception):
    def __init__(self, kind: str, message: str, event: dict[str, Any]):
        super().__init__(message)
        self.kind, self.event = kind, event


class Probe:
    def __init__(self, argv: list[str], name: str, folder: Path):
        self.name = name
        self.stderr_path = folder / f"{name}.stderr.log"
        self.stderr = self.stderr_path.open("xb")
        try:
            self.process = subprocess.Popen(
                argv,
                cwd=ROOT,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.stderr,
                bufsize=0,
            )
        except BaseException:
            self.stderr.close()
            raise
        assert self.process.stdin is not None and self.process.stdout is not None
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)
        self.pending = bytearray()

    def exchange(self, request: dict[str, Any] | None, timeout_ns: int) -> dict[str, Any]:
        started = time.monotonic_ns()
        deadline = started + timeout_ns
        event: dict[str, Any] = {
            "process": self.name,
            "pid": self.process.pid,
            "request": request,
            "send_start_ns": started,
            "hard_deadline_ns": deadline,
            "hard_budget_ns": timeout_ns,
        }
        payload = memoryview(canonical(request) if request is not None else b"")
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdout, selectors.EVENT_READ)
                if payload:
                    selector.register(self.process.stdin, selectors.EVENT_WRITE)
                while b"\n" not in self.pending:
                    remaining = deadline - time.monotonic_ns()
                    if remaining <= 0:
                        raise TimeoutError("request exceeded its monotonic hard deadline")
                    ready = selector.select(remaining / 1_000_000_000)
                    if not ready:
                        raise TimeoutError("request exceeded its monotonic hard deadline")
                    for key, _mask in ready:
                        if key.fileobj is self.process.stdin:
                            try:
                                written = os.write(self.process.stdin.fileno(), payload)
                            except BlockingIOError:
                                continue
                            payload = payload[written:]
                            if not payload:
                                selector.unregister(self.process.stdin)
                                event["request_sent_ns"] = time.monotonic_ns()
                        else:
                            chunk = os.read(self.process.stdout.fileno(), 65536)
                            if not chunk:
                                raise EOFError("player exited before a complete response")
                            self.pending.extend(chunk)
                            if len(self.pending) > RESPONSE_LIMIT:
                                raise ValueError("response exceeded byte limit")
            line, _separator, tail = self.pending.partition(b"\n")
            self.pending = bytearray(tail)
            completed = time.monotonic_ns()
            event.update(
                response_end_ns=completed,
                elapsed_ns=completed - started,
                hard_compliant=completed <= deadline,
                response_sha256=hashlib.sha256(line + b"\n").hexdigest(),
            )
            if payload or self.pending:
                raise ValueError("unsolicited or out-of-order player response")
            event["response_line"] = line.decode(errors="replace")
            event["response"] = json.loads(
                line, object_pairs_hook=unique_keys, parse_constant=reject_constant
            )
            if not isinstance(event["response"], dict):
                raise ValueError("response is not a JSON object")
            return event
        except (OSError, EOFError, TimeoutError, ValueError) as error:
            ended = time.monotonic_ns()
            event.update(
                response_end_ns=ended,
                elapsed_ns=ended - started,
                hard_compliant=False,
                error=str(error),
                partial_stdout=self.pending.decode(errors="replace"),
                process_returncode=self.process.poll(),
            )
            kind = "deadline" if isinstance(error, TimeoutError) else "invalid_or_crashed"
            raise ProbeError(kind, str(error), event) from error

    def close(self) -> dict[str, Any]:
        shutdown = "stdin_eof"
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            shutdown = "terminated_by_driver"
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                shutdown = "killed_by_driver"
                self.process.kill()
                self.process.wait(timeout=1)
        if self.process.stdout is not None:
            self.process.stdout.close()
        self.stderr.close()
        return {
            "name": self.name,
            "pid": self.process.pid,
            "returncode": self.process.returncode,
            "shutdown": shutdown,
            "stderr_path": str(self.stderr_path.relative_to(ROOT)),
            "stderr_sha256": sha(self.stderr_path),
            "stderr_bytes": self.stderr_path.stat().st_size,
            "stderr_preview": self.stderr_path.read_bytes()[:4096].decode(errors="replace"),
        }


def validate_response(response: dict[str, Any], *, control: bool, nodes: int | None = None) -> None:
    if (
        response.get("schema") != "open_shogiai_core_probe/v1"
        or response.get("leaf_sha256") != LEAF_SHA
    ):
        raise ValueError("wrong current runtime schema or leaf identity")
    proof = response["proof"]
    if not isinstance(proof, dict):
        raise ValueError("pure proof is not an object")
    if (
        proof.get("profile") != "pure_learned"
        or proof.get("model_sha256") != LEAF_SHA
        or proof.get("profile_schema") != "open_shogiai_pure_learned_v3_profile/v1"
        or proof.get("evaluator_profile_schema_hash")
        != "d2eec27887926ccc8a076552815cd54e34b85d6d23e65732ddba4989bf59c1e7"
        or any(type(proof.get(key)) is not int or proof[key] != 0 for key in FORBIDDEN)
        or type(proof.get("learned_eval_calls")) is not int
        or proof["learned_eval_calls"] < 0
    ):
        raise ValueError("missing or invalid pure counters")
    if type(response.get("nodes")) is not int or response["nodes"] < 0:
        raise ValueError("invalid node counter")
    if nodes is not None and response["nodes"] > nodes:
        raise ValueError("node budget exceeded")
    if response.get("termination") == "EvaluationError":
        raise ValueError("strict inference failed")
    computation = response.get("compute_control")
    if computation is None and not control:
        return
    if not isinstance(computation, dict):
        raise ValueError("missing computation-control evidence")
    if computation.get("enabled") is not control or computation.get("modelSha256") != CONTROL_SHA:
        raise ValueError("computation-control switch or identity mismatch")


def adjudicated(response: dict[str, Any]) -> dict[str, Any] | None:
    end = response.get("game_end")
    if end == "None":
        if response.get("legal") is not True or not isinstance(response.get("best_move"), str):
            raise ValueError("nonterminal replay has no legal response")
        return None
    if response.get("best_move") is not None or response.get("legal") is not False:
        raise ValueError("terminal replay incorrectly returns a move")
    if response["proof"]["learned_eval_calls"] != 0 or response["nodes"] != 0:
        raise ValueError("terminal adjudication performed evaluation/search")
    if end == "Some(Repetition(NoContest))":
        return {"reason": "repetition", "winner": None}
    match = re.fullmatch(r"Some\(Checkmate \{ winner: (Black|White) \}\)", str(end))
    if match:
        return {"reason": "checkmate", "winner": match[1].lower()}
    match = re.fullmatch(r"Some\(NoLegalMoves \{ loser: (Black|White) \}\)", str(end))
    if match:
        return {"reason": "no_legal_moves", "winner": opposite(match[1].lower())}
    match = re.fullmatch(r"Some\(Repetition\(PerpetualCheckLoss\((Black|White)\)\)\)", str(end))
    if match:
        return {"reason": "perpetual_check", "winner": opposite(match[1].lower())}
    raise ValueError(f"unrecognized terminal result: {end}")


def opposite(side: str) -> str:
    return "white" if side == "black" else "black"


def record_event(game: dict[str, Any], event: dict[str, Any], journal: Any, **fields: Any) -> None:
    event.update(fields)
    game["_last_event"] = event
    game["event_count"] += 1
    journal.write(canonical({"game_id": game["id"], **event}))
    journal.flush()
    if event["kind"] != "move":
        return
    variant = "on" if game["control_by_side"][event["side"]] else "off"
    metrics = game["by_control"].setdefault(
        variant,
        {
            "requests": 0,
            "nodes": 0,
            "learned_eval_calls": 0,
            "charged_elapsed_ns": 0,
            "sum_depth": 0,
            "decisions": 0,
            "reordered_moves": 0,
            "hard_breaches": 0,
        },
    )
    metrics["requests"] += 1
    metrics["charged_elapsed_ns"] += event["elapsed_ns"]
    metrics["hard_breaches"] += int(not event["hard_compliant"])
    response = event.get("response", {})
    proof = response.get("proof")
    proof = proof if isinstance(proof, dict) else {}
    computation = response.get("compute_control")
    computation = computation if isinstance(computation, dict) else {}
    for target, value in {
        "nodes": response.get("nodes"),
        "sum_depth": response.get("depth"),
        "learned_eval_calls": proof.get("learned_eval_calls"),
        "decisions": computation.get("decisions"),
        "reordered_moves": computation.get("reorderedMoves"),
    }.items():
        if type(value) is int and value >= 0:
            metrics[target] += value


def play_game(index: int, argv: list[str], folder: Path, journal: Any) -> dict[str, Any]:
    players: dict[str, Probe] = {}
    side_control = {"black": index == 0, "white": index != 0}
    remaining = {"black": CLOCK_NS, "white": CLOCK_NS}
    game: dict[str, Any] = {
        "id": f"pair-{index + 1}",
        "control_by_side": side_control,
        "initial_sfen": STARTPOS,
        "event_count": 0,
        "by_control": {},
        "moves": [],
        "status": "running",
        "score_control_on": None,
        "remaining_ns": remaining,
    }
    before_cpu, began = cpu_usage(), time.monotonic_ns()
    active_side = "black"
    try:
        for variant in ("on", "off"):
            player = Probe(argv, f"game-{index + 1}-{variant}", folder)
            players[variant] = player
            event = player.exchange(
                {"depth": 1, "nodes": 0, "control": variant == "on"}, 30_000_000_000
            )
            record_event(game, event, journal, kind="startup", charged_to_clock=False)
            validate_response(event["response"], control=variant == "on", nodes=0)
            if adjudicated(event["response"]) is not None:
                raise ValueError("startpos was adjudicated terminal")
        expected_sfen = STARTPOS
        for ply in range(MAX_PLIES):
            active_side = "black" if ply % 2 == 0 else "white"
            enabled = side_control[active_side]
            player = players["on" if enabled else "off"]
            clock_before = remaining.copy()
            request = {
                "moves": game["moves"].copy(),
                "depth": DEPTH,
                "black_time_ms": remaining["black"] // 1_000_000,
                "white_time_ms": remaining["white"] // 1_000_000,
                "control": enabled,
            }
            try:
                event = player.exchange(request, remaining[active_side])
            except ProbeError as error:
                event = error.event
                remaining[active_side] -= event["elapsed_ns"]
                record_event(
                    game,
                    event,
                    journal,
                    kind="move",
                    ply=ply,
                    side=active_side,
                    charged_to_clock=True,
                    clock_before_ns=clock_before,
                    clock_after_ns=remaining.copy(),
                )
                if error.kind == "deadline":
                    game.update(
                        status="completed", reason="time_forfeit", winner=opposite(active_side)
                    )
                    break
                raise
            remaining[active_side] -= event["elapsed_ns"]
            record_event(
                game,
                event,
                journal,
                kind="move",
                ply=ply,
                side=active_side,
                charged_to_clock=True,
                clock_before_ns=clock_before,
                clock_after_ns=remaining.copy(),
            )
            if not event["hard_compliant"] or remaining[active_side] <= 0:
                game.update(status="completed", reason="time_forfeit", winner=opposite(active_side))
                break
            response = event["response"]
            validate_response(response, control=enabled)
            if (
                response.get("sfen") != expected_sfen
                or response.get("perspective", "").lower() != active_side
            ):
                raise ValueError("player searched the wrong root")
            if adjudicated(response) is not None:
                raise ValueError("previous rules-only probe missed a terminal root")
            if response["proof"]["learned_eval_calls"] == 0 and not (
                response["nodes"] == 0
                and response["termination"] in {"TimeLimit", "NodeLimit", "Cancelled"}
            ):
                raise ValueError("nonterminal result lacks required evaluation")
            game["moves"].append(response["best_move"])
            event = players["off"].exchange(
                {"moves": game["moves"].copy(), "depth": 1, "nodes": 0, "control": False},
                5_000_000_000,
            )
            record_event(
                game,
                event,
                journal,
                kind="adjudication",
                ply=ply + 1,
                charged_to_clock=False,
                budget_purpose="rules replay only; zero search nodes",
            )
            validate_response(event["response"], control=False, nodes=0)
            terminal = adjudicated(event["response"])
            if terminal is not None:
                game.update(status="completed", **terminal)
                break
            expected_sfen = event["response"]["sfen"]
        else:
            game.update(status="incomplete", reason="max_plies_unscored", winner=None)
    except (ProbeError, OSError, ValueError, KeyError, TypeError) as error:
        if isinstance(error, ProbeError) and game.get("_last_event") is not error.event:
            record_event(game, error.event, journal, kind="failed_exchange", charged_to_clock=False)
        game.update(
            status="invalid",
            reason=type(error).__name__,
            error=str(error),
            failure_side=active_side,
            winner=None,
        )
    finally:
        game["processes"] = [player.close() for player in players.values()]
        if any(
            process["returncode"] != 0 and process["shutdown"] == "stdin_eof"
            for process in game["processes"]
        ):
            game.update(status="invalid", reason="player_process_failure", winner=None)
        game["child_cpu"] = cpu_delta(before_cpu, cpu_usage())
        game["wall_elapsed_ns"] = time.monotonic_ns() - began
        game.pop("_last_event", None)
    if game["status"] == "completed":
        winner = game["winner"]
        game["score_control_on"] = 0.5 if winner is None else float(side_control[winner])
    return game


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute", action="store_true", help="Actually start the bounded probes/games"
    )
    parser.add_argument("--binary", type=Path, default=Path("target/release/examples/core_probe"))
    parser.add_argument("--binary-sha256", help="Expected SHA-256 of the reviewed release binary")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan = {
        "games": 2,
        "binary_sha256": args.binary_sha256,
        "initial_clock_seconds_each": 20,
        "depth": DEPTH,
        "quiescence_depth": 4,
        "transposition_table_mib": 2,
        "fresh_search_engine_each_request": True,
        "fresh_process_pair_each_game": True,
        "start_position": STARTPOS,
        "color_reversal": True,
        "max_plies": MAX_PLIES,
        "startup_and_zero_node_adjudication_charged_to_clock": False,
        "move_parse_replay_setup_search_transport_charged_to_clock": True,
        "score_incomplete_or_invalid_games": False,
        "retry_or_exclude_failed_games": False,
        "comparison_baseline": (
            "same frozen W256 leaf on repaired common runtime with controller off"
        ),
        "historical_results_reused": False,
        "strength_claim": False,
        "leaf_sha256": LEAF_SHA,
        "controller_sha256": CONTROL_SHA,
    }
    if not args.execute:
        print(json.dumps({"status": "prepared_not_executed", "plan": plan}, indent=2))
        return
    if not args.binary_sha256 or not re.fullmatch(r"[0-9a-f]{64}", args.binary_sha256):
        parser.error("--execute requires a lowercase --binary-sha256")
    output = args.output or OUTPUT_ROOT / datetime.now(UTC).strftime("run-%Y%m%dT%H%M%S%fZ")
    output = output if output.is_absolute() else ROOT / output
    if any(
        path.is_symlink() for path in [output, *output.parents]
    ) or not output.resolve().is_relative_to(OUTPUT_ROOT):
        raise ValueError("output must stay inside local/core-prototype/comparison without symlinks")
    output.mkdir(parents=True, exist_ok=False)
    receipt: dict[str, Any] = {
        "schema": "open_shogiai_core_comparison/v1",
        "plan": plan,
        "games": [],
        "summary": {"planned_games": 2, "scheduled_games": 0, "recorded_games": 0},
        "started_utc": datetime.now(UTC).isoformat(),
    }
    try:
        leaf = artifact(Path("local/frozen/baseline/model.osaval03"), LEAF_SHA)
        controller = artifact(Path("local/core-prototype/controller.json"), CONTROL_SHA)
        binary = artifact(args.binary, args.binary_sha256)
        frozen_binary = snapshot(binary, output, "core_probe")
        frozen_controller = snapshot(controller, output, "controller.json")
        receipt["artifacts"] = {
            "leaf": leaf,
            "current_binary": binary,
            "current_binary_snapshot": frozen_binary,
            "controller": controller,
            "controller_snapshot": frozen_controller,
        }
        current_argv = [
            str(ROOT / frozen_binary["path"]),
            str(ROOT / leaf["path"]),
            LEAF_SHA,
            "probe",
            str(ROOT / frozen_controller["path"]),
            CONTROL_SHA,
        ]
        receipt["source_observation"] = {
            "head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "files": {
                name: sha(ROOT / name)
                for name in (
                    "engine/core/examples/core_probe.rs",
                    "engine/core/src/search.rs",
                    "engine/core/src/computation.rs",
                    "engine/core/src/game.rs",
                    "engine/core/src/runtime_profile.rs",
                    "scripts/compare_core_prototype.py",
                )
            },
            "note": "working-tree observation; does not itself attest a reproducible binary build",
        }
        journal = TraceJournal(output / "events.jsonl.gz")
        receipt["summary"]["scheduled_games"] = 2
        try:
            for index in range(2):
                game = play_game(index, current_argv, output, journal)
                receipt["games"].append(game)
                receipt["summary"]["recorded_games"] = len(receipt["games"])
                print(
                    json.dumps(
                        {
                            "game": game["id"],
                            "status": game["status"],
                            "reason": game.get("reason"),
                            "plies": len(game["moves"]),
                        }
                    ),
                    flush=True,
                )
        finally:
            receipt["trace"] = journal.close()
        for ref in (leaf, frozen_binary, frozen_controller):
            artifact(Path(ref["path"]), ref["sha256"])
        receipt["cleanup"] = [remove_duplicate(frozen_controller, controller)]
        games = receipt["games"]
        complete = len(games) == 2 and all(game["status"] == "completed" for game in games)
        receipt["summary"] = {
            "planned_games": 2,
            "scheduled_games": 2,
            "recorded_games": len(games),
            "completed_games": sum(game["status"] == "completed" for game in games),
            "invalid_games": sum(game["status"] == "invalid" for game in games),
            "incomplete_games": sum(game["status"] == "incomplete" for game in games),
            "control_on_score_over_all_two_games": sum(game["score_control_on"] for game in games)
            / 2
            if complete
            else None,
            "not_a_strength_estimate": True,
        }
        receipt["status"] = "finished"
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt["completed_utc"] = datetime.now(UTC).isoformat()
        receipt["receipt_sha256"] = hashlib.sha256(canonical(receipt)).hexdigest()
        with (output / "receipt.json").open("xb") as stream:
            stream.write(canonical(receipt))
        print(
            json.dumps(
                {
                    "receipt": str((output / "receipt.json").relative_to(ROOT)),
                    "status": receipt["status"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
