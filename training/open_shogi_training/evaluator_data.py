"""Resumable, game-partitioned teacher trajectories for the current evaluator run.

The teacher is offline only. Every move is replayed by our native legal engine.
Raw observations are retained; mate scores never become scalar centipawns.
"""

from __future__ import annotations

import errno
import gzip
import hashlib
import json
import os
import random
import selectors
import subprocess
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import numpy as np

from .evaluator_ledger import (
    DeferredTaskError,
    Ledger,
    export_queue,
    pack_result,
    restore_rng,
    task_identity,
    unpack_result,
)
from .labeling.config import load_teacher_config
from .labeling.usi import USIEngine, USIError, USIIncompleteDepthError, USITerminalResult
from .phase10r_model import parse_sfen
from .phase10v_data import position_hash
from .phase10v_model import sparse_features

START = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
SCHEMA = "open_shogiai_evaluator_data/v1"
MAX_FEATURES = 48


def encoded(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    for attempt in range(3):
        try:
            with temporary.open("wb") as f:
                f.write(value)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, path)
            return
        except OSError as error:
            if error.errno not in (errno.EAGAIN, errno.EINTR, errno.EBUSY) or attempt == 2:
                raise
            time.sleep(0.05 * (attempt + 1))


def partition(game: int, seed: int) -> str:
    # Assigned from the original trajectory identity, before labels/features/normalization.
    bucket = int(hashlib.sha256(f"trajectory:{seed}:{game}".encode()).hexdigest()[:8], 16) % 20
    return "validation" if bucket == 0 else "development_test" if bucket == 1 else "train"


def symmetry_keys(sfen: str) -> set[str]:
    """Canonical SFEN hashes including left/right and color/180-degree symmetries."""
    p = parse_sfen(sfen)
    names = ("P", "L", "N", "S", "G", "B", "R", "K", "+P", "+L", "+N", "+S", "+B", "+R")
    keys = set()
    for rotate in (False, True):
        for mirror in (False, True):
            board = [None] * 81
            for piece in p.board:
                if piece is None:
                    continue
                square = 80 - piece.square if rotate else piece.square
                rank, file = divmod(square, 9)
                square = rank * 9 + (8 - file if mirror else file)
                name = names[piece.kind]
                board[square] = name.lower() if piece.side ^ rotate else name
            ranks = []
            for rank in range(9):
                text, empty = "", 0
                for piece in board[rank * 9 : rank * 9 + 9]:
                    if piece is None:
                        empty += 1
                    else:
                        text += (str(empty) if empty else "") + piece
                        empty = 0
                ranks.append(text + (str(empty) if empty else ""))
            hand = ""
            for side in (0, 1):
                for name in "RBGSNLP":
                    count = p.hands[side ^ rotate]["PLNSGBR".index(name)]
                    if count:
                        hand += (str(count) if count > 1 else "") + (name.lower() if side else name)
            state = f"{'/'.join(ranks)} {'w' if p.side_to_move ^ rotate else 'b'} {hand or '-'}"
            keys.add(hashlib.sha256(state.encode()).hexdigest())
    return keys


class Replay:
    """One owned native process, bounded requests and no silent rules fallback."""

    def __init__(self, root: Path, config: dict):
        self.process = subprocess.Popen(
            [
                str(root / config["replay_path"]),
                str(root / config["leaf_path"]),
                config["leaf_sha256"],
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)

    def ask(self, **request: object) -> dict:
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        if not self.selector.select(30):
            raise RuntimeError("native replay made no progress for 30 seconds")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("native replay exited: " + self.process.stderr.read()[-2000:])
        return json.loads(line)

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=2)
        self.selector.close()


def teacher_config(root: Path, config: dict):
    original = load_teacher_config(root / config["teacher_config_path"])
    return replace(
        original,
        timeouts=replace(
            original.timeouts,
            ready_ms=int(
                config.get("recovery_policy", {}).get(
                    "ready_seconds", original.timeouts.ready_ms / 1000
                )
                * 1000
            ),
            startup_ms=int(
                config.get("recovery_policy", {}).get(
                    "startup_seconds", original.timeouts.startup_ms / 1000
                )
                * 1000
            ),
            stop_ms=int(
                config.get("recovery_policy", {}).get(
                    "stop_seconds", original.timeouts.stop_ms / 1000
                )
                * 1000
            ),
            quit_ms=int(
                config.get("recovery_policy", {}).get(
                    "quit_seconds", original.timeouts.quit_ms / 1000
                )
                * 1000
            ),
            search_ms=int(
                config.get("recovery_policy", {}).get(
                    "maximum_search_seconds", original.timeouts.search_ms / 1000
                )
                * 1000
            ),
        ),
        binary_sha256=config["teacher_binary_sha256"],
        threads=1,
        hash_mb=64,
        multipv=3,
        nodes=config["teacher_nodes"],
        options=(
            ("Book_Enable", False),
            ("Eval_Dir", "eval/20190617"),
            ("Eval_Hash", 16),
            ("USI_Ponder", False),
        ),
    )


def _validated_teacher_terminal(result: USITerminalResult, state: dict, ply: int) -> dict:
    side = "black" if state["sfen"].split()[1] == "b" else "white"
    declaration = state.get("teacher_declaration")
    if result.outcome == "win":
        required = 28 if side == "black" else 27
        if (
            state["terminal"] != "None"
            or not isinstance(declaration, dict)
            or declaration.get("rule") != "csa_28_27"
            or declaration.get("side") != side
            or declaration.get("minimum_camp_pieces") != 10
            or declaration.get("required_points") != required
            or declaration.get("result") != "win"
            or type(declaration.get("points")) is not int
            or declaration["points"] < required
        ):
            raise ValueError("teacher win is not a verified native CSA declaration")
        validation = "native_csa_28_27"
    elif result.outcome == "resign":
        if state["terminal"] == "None" or state.get("teacher_resign_eligible") is not True:
            raise ValueError("teacher resign disagrees with native no-legal-move termination")
        validation = "native_no_legal_moves"
    else:
        raise ValueError("unknown teacher terminal outcome")
    if result.raw_bestmove != f"bestmove {result.outcome}":
        raise ValueError("teacher terminal bestmove is not the exact two-token response")
    return {
        "kind": "terminal",
        "outcome": result.outcome,
        "validation": validation,
        "raw_bestmove": result.raw_bestmove,
        "sfen": state["sfen"],
        "ply": ply,
        "native_terminal": state["terminal"],
        "native_declaration": declaration,
    }


def _ledger_observation(teacher, state, *, root, output, config, game, ply, branch, moves, records):
    from .labeling.usi import USIProcessError, USIProtocolError, USIResourceError, USITimeoutError

    ledger = Ledger(output)
    key, saved = ledger.task(task_identity(config, game, ply, branch, state, moves))
    try:
        if saved["status"] == "accepted":
            result = unpack_result(saved["result"])
            return result, _validated_teacher_terminal(result, state, ply) if isinstance(
                result, USITerminalResult
            ) else None
        if saved["status"] == "deferred":
            raise DeferredTaskError(key)
        attempts = len(saved["attempts"])
        if attempts >= 2:
            ledger.finish(key, "deferred")
            raise DeferredTaskError(key)
        nodes = (
            config["teacher_retry_nodes"]
            if any(a["outcome"] == "USIIncompleteDepthError" for a in saved["attempts"])
            else config["teacher_nodes"]
        )
        try:
            ledger.begin(key, nodes, policy=config["recovery_policy"])
        except DeferredTaskError as error:
            raise DeferredTaskError(key) from error
        began = time.monotonic()
        try:
            result = teacher.analyze(
                state["sfen"],
                nodes=nodes,
                depth=config["teacher_depth"],
                expected_candidates=min(3, len(state["successors"])),
                auto_start=False,
            )
            terminal = (
                _validated_teacher_terminal(result, state, ply)
                if isinstance(result, USITerminalResult)
                else None
            )
            if not terminal:
                legal = {child["move"] for child in state["successors"]}
                if any(c.pv[0] not in legal for c in result.candidates):
                    raise ValueError("teacher proposed illegal move")
            ledger.finish(
                key,
                "accepted",
                result=pack_result(result),
                evidence={"outcome": "accepted", "elapsed_s": time.monotonic() - began},
            )
            return result, terminal
        except USIResourceError:
            raise
        except (USIIncompleteDepthError, USIProcessError, USITimeoutError) as error:
            status = "hard" if attempts == 0 else "deferred"
            ledger.finish(
                key,
                status,
                evidence={
                    "outcome": type(error).__name__,
                    "elapsed_s": time.monotonic() - began,
                    "stdout_tail": getattr(error, "stdout_tail", "")[-16384:],
                    "stderr_tail": getattr(error, "stderr_tail", "")[-8192:],
                    "bestmove": getattr(error, "bestmove_line", None),
                },
            )
            if not isinstance(error, USIIncompleteDepthError):
                teacher.close()
                while True:
                    restarts = ledger.increment("worker_restarts")
                    if restarts > config["recovery_policy"]["maximum_worker_restarts"]:
                        raise USIProcessError("worker recovery budget exhausted") from error
                    try:
                        teacher.start()
                        break
                    except (USIProcessError, USITimeoutError):
                        teacher.close()  # Only typed connection failures consume another slot.
            ledger.check_health()
            raise DeferredTaskError(key) from error
        except (USIProtocolError, ValueError) as error:
            ledger.finish(
                key, "failed", evidence={"outcome": type(error).__name__, "error": str(error)}
            )
            raise
    finally:
        ledger.close()


def _teacher_observation(
    teacher, state, *, root, output, config, game, ply, branch, moves, records
):
    """Keep rejected teacher evidence before propagating its fail-closed exception."""
    if config.get("recovery_policy"):
        return _ledger_observation(
            teacher,
            state,
            root=root,
            output=output,
            config=config,
            game=game,
            ply=ply,
            branch=branch,
            moves=moves,
            records=records,
        )
    try:
        if config.get("teacher_depth") is not None:
            result = teacher.analyze_with_retry(
                state["sfen"], nodes=config["teacher_nodes"], depth=config["teacher_depth"]
            )
        else:
            result = teacher.analyze_with_retry(state["sfen"])
        if isinstance(result, USITerminalResult):
            return result, _validated_teacher_terminal(result, state, ply)
        if "successors" in state:
            legal = {child["move"] for child in state["successors"]}
            if any(candidate.pv[0] not in legal for candidate in result.candidates):
                raise ValueError("teacher proposed illegal move")
        return result, None
    except (USIError, ValueError) as error:
        if not output.resolve().is_relative_to((root / "local").resolve()):
            raise ValueError(
                "teacher failure evidence must stay in ignored local output"
            ) from error
        diagnosis = getattr(teacher, "search_diagnostics", {})
        bestmove_line = getattr(error, "bestmove_line", None) or diagnosis.get("bestmove_line")
        stdout_tail = getattr(error, "stdout_tail", "") or diagnosis.get("stdout_tail", "")
        stem = f"{game:06d}-ply{ply:04d}-{branch}-{uuid.uuid4().hex}"
        prefix = output / "failures" / f"{stem}.prefix.json.gz"
        atomic(prefix, gzip.compress(encoded({"moves": moves, "records": records}), mtime=0))
        receipt = {
            "schema": "open_shogiai_teacher_failure/v1",
            "game": game,
            "ply": ply,
            "branch": branch,
            "sfen": state["sfen"],
            "error_type": type(error).__name__,
            "error": str(error)[:8192],
            "bestmove_line": bestmove_line,
            "stdout_tail": stdout_tail,
            "stderr_tail": getattr(error, "stderr_tail", "")[-8192:],
            "native_terminal": state["terminal"],
            "native_declaration": state.get("teacher_declaration"),
            "native_resign_eligible": state.get("teacher_resign_eligible"),
            "generation_config_sha256": hashlib.sha256(encoded(config)).hexdigest(),
            "completed_records": len(records),
            "prefix": {"path": prefix.name, "sha256": digest(prefix)},
        }
        atomic(prefix.with_name(f"{stem}.json"), encoded(receipt))
        retry_nodes = config.get("teacher_retry_nodes")
        if (
            isinstance(error, USIIncompleteDepthError)
            and config.get("teacher_depth")
            and type(retry_nodes) is int
            and retry_nodes > config["teacher_nodes"]
        ):
            return _teacher_observation(
                teacher,
                state,
                root=root,
                output=output,
                config={**config, "teacher_nodes": retry_nodes, "teacher_retry_nodes": None},
                game=game,
                ply=ply,
                branch=branch,
                moves=moves,
                records=records,
            )
        error.failure_receipt = str(prefix.with_name(f"{stem}.json").relative_to(output))
        error.requested_depth = config.get("teacher_depth")
        error.node_ceiling = config.get("teacher_nodes")
        raise


def _optional_focus_observation(*args, **kwargs):
    """Only an exhausted optional depth budget becomes explicit missing evidence."""
    try:
        return _teacher_observation(*args, **kwargs)
    except DeferredTaskError:
        # Finish the bounded optional candidate group before its immutable shard commit.
        # Both attempts are charged in the same ledger; no outer retry resets them.
        try:
            return _teacher_observation(*args, **kwargs)
        except DeferredTaskError as exhausted:
            ledger = Ledger(kwargs["output"])
            saved = ledger.get(str(exhausted))
            ledger.close()
            return None, {
                "status": "unlabeled_deferred",
                "ledger_task": str(exhausted),
                "failure_kind": saved["attempts"][-1]["outcome"],
                "recovery": "not_attempted",
            }
    except USIIncompleteDepthError as error:
        return None, {
            "status": "unlabeled_incomplete_depth",
            "requested_depth": error.requested_depth,
            "node_ceiling": error.node_ceiling,
            "failure_receipt": error.failure_receipt,
            "failure_sha256": digest(kwargs["output"] / error.failure_receipt),
            "recovery": "not_attempted",
        }


def _terminal_counts(records: list[dict]) -> dict[str, int]:
    counts = {}
    for row in records:
        for branch, observation in (("root", row), ("deviation", row.get("deviation"))):
            terminal = observation.get("terminal_outcome") if observation else None
            if terminal:
                key = f"{branch}_{terminal['outcome']}"
                counts[key] = counts.get(key, 0) + 1
    return counts


def _generate_group(root_name: str, output_name: str, config: dict, games: list[int]) -> list[dict]:
    root, output = Path(root_name), Path(output_name)
    if all((output / "games" / f"{game:06d}.json.receipt.json").exists() for game in games):
        reports = []
        for game in games:
            target = output / "games" / f"{game:06d}.json.gz"
            saved = json.loads(target.with_suffix(".receipt.json").read_text())
            if digest(target) != saved["sha256"]:
                raise ValueError("completed trajectory corrupted")
            reports.append(saved)
        return reports
    reports = []
    unlabeled_in_worker = 0
    replay = Replay(root, config)
    campaign = config.get("defense_campaign")
    probe, branch_replay = None, None
    ledger = Ledger(output) if config.get("recovery_policy") else None
    try:
        if campaign:
            from .defense_scenarios import R3Probe

            probe = R3Probe(root, config)
            branch_replay = Replay(root, config)
        with USIEngine(
            teacher_config(root, config),
            root,
            isolate_process_group=False,
            allow_terminal_outcomes=True,
        ) as teacher:
            for game in games:
                target = output / "games" / f"{game:06d}.json.gz"
                receipt = target.with_suffix(".receipt.json")
                if receipt.exists():
                    saved = json.loads(receipt.read_text())
                    if digest(target) != saved["sha256"]:
                        raise ValueError("completed trajectory corrupted")
                    reports.append(saved)
                    unlabeled_in_worker += saved.get("unlabeled_focus", 0)
                    continue
                rng = random.Random(config["seed"] + game)
                began = time.monotonic()
                initial, prefix, family, variant = START, [], None, None
                if campaign:
                    from .defense_scenarios import assignment, rotate_move, rotate_sfen

                    family, variant = assignment(config, game)
                    initial = rotate_sfen(START) if variant % 2 else START
                    prefix = [rotate_move(m) if variant % 2 else m for m in family["moves"]]
                state = replay.ask(reset=initial, successors=True)
                records, moves, trajectory_outcome = [], [], None
                checkpoint = ledger.load_checkpoint(game) if ledger else None
                first_ply = 0
                if checkpoint:
                    records, moves = checkpoint["records"], checkpoint["moves"]
                    rng.setstate(restore_rng(checkpoint["rng"]))
                    for move in moves:
                        state = replay.ask(movement=move, successors=True)
                    if state["sfen"] != checkpoint["sfen"]:
                        raise ValueError("checkpoint replay/history mismatch")
                    first_ply = checkpoint["ply"]
                deferred = False
                for ply in range(first_ply, config["max_plies"]):
                    if ledger:
                        ledger.checkpoint(
                            game,
                            {
                                "records": records,
                                "moves": moves,
                                "rng": rng.getstate(),
                                "ply": ply,
                                "sfen": state["sfen"],
                                "initial_sfen": initial,
                                "status": "pending",
                            },
                        )
                    if (output / "STOP").exists():
                        raise InterruptedError("requested stop; completed trajectories retained")
                    if state["terminal"] != "None":
                        break
                    try:
                        result, terminal = _teacher_observation(
                            teacher,
                            state,
                            root=root,
                            output=output,
                            config=config,
                            game=game,
                            ply=ply,
                            branch="root",
                            moves=moves,
                            records=records,
                        )
                    except DeferredTaskError:
                        deferred = True
                        break
                    if terminal is not None:
                        records.append(
                            {
                                "sfen": state["sfen"],
                                "ply": ply,
                                "candidates": [],
                                "deviation": None,
                                "terminal_outcome": terminal,
                                "teacher_elapsed_ms": result.elapsed_ms,
                            }
                        )
                        trajectory_outcome = terminal
                        break
                    legal = {c["move"]: c for c in state["successors"]}
                    if ply % config["sample_stride"] == 0:
                        candidates = [
                            dict(
                                c.as_dict(),
                                child_sfen=legal[c.pv[0]]["sfen"],
                                child_terminal=legal[c.pv[0]]["terminal"],
                            )
                            for c in result.candidates
                        ]
                        record = {
                            "sfen": state["sfen"],
                            "ply": ply,
                            "candidates": candidates,
                            "teacher_elapsed_ms": result.elapsed_ms,
                            "deviation": None,
                        }
                        # Offline coverage of the current weak evaluator's actual preferred child.
                        weak = min(legal.values(), key=lambda c: (c["child_cp"], c["move"]))
                        if campaign and ply % campaign["probe_stride"] == 0:
                            observed = probe.search(initial, moves, state["sfen"])
                            if observed["best_move"] not in legal:
                                raise ValueError("r3 diagnostic selected illegal move")
                            weak = legal[observed["best_move"]]
                            record["r3_search"] = observed
                        if (
                            weak["terminal"] == "None"
                            and ply
                            % (campaign["probe_stride"] if campaign else config["deviation_stride"])
                            == 0
                            and (
                                campaign or weak["move"] not in {c.pv[0] for c in result.candidates}
                            )
                        ):
                            if campaign:
                                branch_replay.ask(reset=initial, successors=True)
                                for history_move in moves:
                                    branch_replay.ask(movement=history_move, successors=True)
                                child_state = branch_replay.ask(
                                    movement=weak["move"], successors=True
                                )
                            else:
                                child_state = {**weak, "sfen": weak["sfen"]}
                            child_config = config
                            if campaign:
                                child_config = {
                                    **config,
                                    "teacher_nodes": campaign["relabel_nodes"],
                                    "teacher_depth": campaign["relabel_depth"],
                                }
                            observer = (
                                _optional_focus_observation if campaign else _teacher_observation
                            )
                            child_result, child_terminal = observer(
                                teacher,
                                child_state,
                                root=root,
                                output=output,
                                config=child_config,
                                game=game,
                                ply=ply + 1,
                                branch="deviation",
                                moves=[*moves, weak["move"]],
                                records=[*records, record],
                            )
                            record["deviation"] = {
                                "move": weak["move"],
                                "sfen": weak["sfen"],
                                "teacher_elapsed_ms": child_result.elapsed_ms
                                if child_result
                                else None,
                            }
                            if child_result is None:
                                record["deviation"].update(child_terminal)
                            elif child_terminal is not None:
                                record["deviation"]["terminal_outcome"] = child_terminal
                            else:
                                record["deviation"].update(
                                    score=child_result.primary.score.as_dict(),
                                    candidates=[c.as_dict() for c in child_result.candidates],
                                )
                                if campaign:
                                    reply = child_result.primary.pv[0]
                                    recovery_state = branch_replay.ask(
                                        movement=reply, successors=True
                                    )
                                    if recovery_state["terminal"] == "None":
                                        recovery, recovery_terminal = _optional_focus_observation(
                                            teacher,
                                            recovery_state,
                                            root=root,
                                            output=output,
                                            config=child_config,
                                            game=game,
                                            ply=ply + 2,
                                            branch="recovery",
                                            moves=[*moves, weak["move"], reply],
                                            records=[*records, record],
                                        )
                                        record["deviation"]["recovery"] = {
                                            "sfen": recovery_state["sfen"],
                                            "reply": reply,
                                            "score": (
                                                recovery_terminal
                                                if recovery_terminal is not None
                                                else recovery.primary.score.as_dict()
                                            ),
                                            "candidates": (
                                                []
                                                if recovery_terminal is not None
                                                else [c.as_dict() for c in recovery.candidates]
                                            ),
                                            "teacher_elapsed_ms": recovery.elapsed_ms
                                            if recovery
                                            else None,
                                        }
                                        if recovery is None:
                                            record["deviation"]["recovery"] = {
                                                "sfen": recovery_state["sfen"],
                                                "reply": reply,
                                                **{
                                                    k: v
                                                    for k, v in recovery_terminal.items()
                                                    if k != "recovery"
                                                },
                                            }
                        records.append(record)
                        if campaign:
                            d = record.get("deviation")
                            missing = sum(
                                isinstance(observation, dict)
                                and observation.get("status")
                                in ("unlabeled_incomplete_depth", "unlabeled_deferred")
                                for observation in (d, d.get("recovery") if d else None)
                            )
                            unlabeled_in_worker += missing
                            if (
                                not ledger
                                and unlabeled_in_worker > campaign["maximum_unlabeled_focus"]
                            ):
                                atomic(
                                    output / f"focus-limit-{games[0]}.json",
                                    encoded(
                                        {
                                            "status": "needs_astra",
                                            "game": game,
                                            "ply": ply,
                                            "unlabeled_in_worker": unlabeled_in_worker,
                                            "maximum_unlabeled_focus": campaign[
                                                "maximum_unlabeled_focus"
                                            ],
                                        }
                                    ),
                                )
                                raise ValueError(
                                    "optional focus absolute missing-label ceiling exceeded"
                                )
                    # Data generation only: varied strong continuations, never a runtime preset.
                    weights = [0.6, 0.3, 0.1] if ply < 48 else [0.92, 0.06, 0.02]
                    choice = rng.choices(result.candidates, weights[: len(result.candidates)])[
                        0
                    ].pv[0]
                    if ply < len(prefix):
                        choice = prefix[ply]
                        if choice not in legal:
                            raise ValueError("offline prefix no longer legal")
                    moves.append(choice)
                    state = replay.ask(movement=choice, successors=True)
                if deferred:
                    continue
                raw = {
                    "schema": SCHEMA,
                    "game": game,
                    "seed": config["seed"] + game,
                    "split": partition(game, config["seed"]),
                    "moves": moves,
                    "end": state["terminal"] if trajectory_outcome is None else "TeacherTerminal",
                    "records": records,
                }
                if campaign:
                    raw.update(
                        family=family["id"],
                        variant=variant,
                        group=family["group"],
                        split=family["split"],
                        prefix_moves=prefix,
                        initial_sfen=initial,
                        source="generated",
                    )
                if trajectory_outcome is not None:
                    raw["terminal_outcome"] = trajectory_outcome
                # An interrupted old temporary output is replaced, never a completed receipt.
                atomic(target, gzip.compress(encoded(raw), mtime=0))
                saved = {
                    "game": game,
                    "sha256": digest(target),
                    "rows": len(records),
                    "plies": len(moves),
                    "elapsed_s": time.monotonic() - began,
                    "split": raw["split"],
                    "teacher_terminal_outcomes": _terminal_counts(records),
                }
                if campaign:
                    canonical_moves = [rotate_move(m) if variant % 2 else m for m in moves]
                    saved.update(
                        family=family["id"],
                        group=family["group"],
                        variant=variant,
                        trajectory_sha256=hashlib.sha256(
                            encoded([family["id"], canonical_moves])
                        ).hexdigest(),
                        unlabeled_focus=sum(
                            isinstance(observation, dict)
                            and observation.get("status")
                            in ("unlabeled_incomplete_depth", "unlabeled_deferred")
                            for r in records
                            for d in [r.get("deviation")]
                            for observation in (d, d.get("recovery") if d else None)
                        ),
                        candidate_examples=sum(len(r["candidates"]) for r in records),
                        r3_searches=sum("r3_search" in r for r in records),
                        deviation_labels=sum(r.get("deviation") is not None for r in records),
                        recovery_labels=sum(
                            isinstance(r.get("deviation", {}).get("recovery"), dict)
                            for r in records
                            if r.get("deviation")
                        ),
                    )
                atomic(receipt, encoded(saved))
                reports.append(saved)
                atomic(output / f"worker-{games[0]}.json", encoded(saved))
    finally:
        replay.close()
        if ledger:
            ledger.close()
        if probe is not None:
            probe.close()
        if branch_replay is not None:
            branch_replay.close()
    return reports


def _drain_hard_queue(root: Path, output: Path, config: dict):
    for _round in range(config["max_plies"]):
        ledger = Ledger(output)
        pending = list(
            ledger.db.execute(
                "SELECT DISTINCT json_extract(identity,'$.game') FROM tasks "
                "WHERE status='hard' AND json_extract(identity,'$.branch')='root'"
            )
        )
        ledger.close()
        if not pending:
            break
        # Interleave defense/attack with other groups, retaining ascending order in each bucket.
        priority, ordinary = [], []
        for (game,) in pending:
            from .defense_scenarios import assignment

            family, _ = assignment(config, game)
            (priority if family["group"] in ("defense", "attack_end") else ordinary).append(game)
        hard_games = []
        while priority or ordinary:
            if priority:
                hard_games.append(priority.pop(0))
            if ordinary:
                hard_games.append(ordinary.pop(0))
        for game in hard_games:
            _generate_group(str(root), str(output), config, [game])


def generate(
    root: Path,
    output: Path,
    config: dict,
    *,
    pause_after_new_games: int | None = None,
    supplemental: bool = False,
) -> dict:
    for path_key, sha_key in [
        ("replay_path", "replay_sha256"),
        ("leaf_path", "leaf_sha256"),
        ("teacher_config_path", "teacher_config_sha256"),
    ]:
        if digest(root / config[path_key]) != config[sha_key]:
            raise ValueError(f"identity changed: {path_key}")
    output.mkdir(parents=True, exist_ok=True)
    if config.get("defense_campaign"):
        from .defense_scenarios import verify_prefixes

        atomic(output / "prefix-validation.json", encoded(verify_prefixes(root, config)))
    identity = output / "generation.json"
    if identity.exists() and identity.read_bytes() != encoded(config):
        raise ValueError("generation config changed; new run required")
    atomic(identity, encoded(config))
    start = config.get("first_game", 0)
    limit = config["games"] + config.get("recovery_policy", {}).get("maximum_supplemental_games", 0)
    expected = {f"{game:06d}.json.gz" for game in range(start, start + limit)}
    if any(p.name not in expected for p in (output / "games").glob("*.json.gz")):
        raise ValueError("unexpected trajectory outside the sealed generation range")
    if config.get("recovery_policy"):
        all_reports = []
        new_games = 0
        # Main tasks get one finite pass before hard tasks; every queued game gets a turn.
        ranges = [list(range(start, start + config["games"]))]
        for games in ranges:
            for game in games:
                present = (output / "games" / f"{game:06d}.json.receipt.json").exists()
                reports = _generate_group(str(root), str(output), config, [game])
                all_reports.extend(reports)
                new_games += bool(reports) and not present
                if pause_after_new_games is not None and new_games >= pause_after_new_games:
                    ledger = Ledger(output)
                    progress = {
                        "status": "paused",
                        "games": len(list((output / "games").glob("*.receipt.json"))),
                        "tasks": ledger.summary(),
                    }
                    ledger.close()
                    export_queue(output)
                    atomic(output / "generation-progress.json", encoded(progress))
                    return progress
        _drain_hard_queue(root, output, config)
        from .evaluator_coverage import coverage_report

        export_queue(output)
        coverage = coverage_report(output, config)
        if supplemental or not coverage.get("passed", coverage.get("eligible", False)):
            supplemental = config["recovery_policy"].get("maximum_supplemental_games", 192)
            for game in range(start + config["games"], start + config["games"] + supplemental):
                _generate_group(str(root), str(output), config, [game])
            _drain_hard_queue(root, output, config)
            export_queue(output)
            coverage = coverage_report(output, config)
        all_reports = [
            json.loads(p.read_text()) for p in sorted((output / "games").glob("*.receipt.json"))
        ]
    else:
        groups = [
            list(range(start + i, start + config["games"], config["workers"]))
            for i in range(config["workers"])
        ]
        all_reports = []
        with ProcessPoolExecutor(max_workers=config["workers"]) as pool:
            futures = [
                pool.submit(_generate_group, str(root), str(output), config, g) for g in groups if g
            ]
            for future in as_completed(futures):
                try:
                    all_reports.extend(future.result())
                except BaseException:
                    (output / "STOP").touch()
                    raise
    report = {
        "schema": SCHEMA,
        "games": len(all_reports),
        "sampled_roots": sum(r["rows"] for r in all_reports),
        "plies": sum(r["plies"] for r in all_reports),
        "config_sha256": digest(identity),
        "teacher_terminal_outcomes": {
            key: sum(r.get("teacher_terminal_outcomes", {}).get(key, 0) for r in all_reports)
            for key in ("root_win", "root_resign", "deviation_win", "deviation_resign")
        },
    }
    if config.get("defense_campaign"):
        from .defense_scenarios import focus_gate

        report["focus_quality"] = focus_gate(output, config)
        report.update(
            scenario_families=len({r["family"] for r in all_reports}),
            unique_trajectory_paths=len({r["trajectory_sha256"] for r in all_reports}),
            family_splits={
                s: sorted({r["family"] for r in all_reports if r["split"] == s})
                for s in ("train", "validation", "development_test")
            },
            prefix_validation_sha256=digest(output / "prefix-validation.json"),
            **{
                k: sum(r[k] for r in all_reports)
                for k in (
                    "candidate_examples",
                    "r3_searches",
                    "deviation_labels",
                    "recovery_labels",
                )
            },
            independence_limit=(
                "Color/stochastic variants are clustered by original scenario family; "
                "they are not independent human games."
            ),
        )
    if config.get("recovery_policy"):
        report["coverage"] = coverage
        report["status"] = (
            "complete"
            if coverage.get("passed", coverage.get("eligible", False))
            else "coverage_exhausted"
        )
        if report["status"] != "complete":
            atomic(output / "generation-progress.json", encoded(report))
            return report
    elif {r["game"] for r in all_reports} != set(range(start, start + config["games"])):
        raise ValueError("generation did not complete its exact trajectory set")
    atomic(output / "generation-complete.json", encoded(report))
    return report


def _observations(game: dict):
    for index, row in enumerate(game["records"]):
        if row.get("terminal_outcome") is not None:
            yield row["sfen"], row["terminal_outcome"], "root", row["ply"], index, None
            continue
        primary = row["candidates"][0]
        yield row["sfen"], primary["score"], "root", row["ply"], index, None
        deviation = row.get("deviation")
        recovery = deviation.get("recovery") if deviation else None
        masked = {
            position_hash(observation["sfen"])
            for observation in (deviation, recovery)
            if isinstance(observation, dict)
            and observation.get("status") in ("unlabeled_incomplete_depth", "unlabeled_deferred")
        }
        for c in row["candidates"]:
            if c["child_terminal"] != "None" or position_hash(c["child_sfen"]) in masked:
                continue
            yield (
                c["child_sfen"],
                {"kind": c["score"]["kind"], "value": -c["score"]["value"]},
                "candidate",
                row["ply"] + 1,
                index,
                c["pv"][0],
            )
        if row["deviation"] and row["deviation"].get("status") not in (
            "unlabeled_incomplete_depth",
            "unlabeled_deferred",
        ):
            d = row["deviation"]
            score = d["terminal_outcome"] if d.get("terminal_outcome") is not None else d["score"]
            yield d["sfen"], score, "deviation", row["ply"] + 1, index, d["move"]
            if isinstance(d.get("recovery"), dict) and d["recovery"].get("status") not in (
                "unlabeled_incomplete_depth",
                "unlabeled_deferred",
            ):
                r = d["recovery"]
                yield r["sfen"], r["score"], "recovery", row["ply"] + 2, index, r["reply"]


def prepare(root: Path, output: Path, config: dict, excluded_sfens: list[str]) -> dict:
    if config.get("defense_campaign"):
        from .defense_scenarios import prepare_campaign

        return prepare_campaign(root, output, config, excluded_sfens)
    return _prepare(root, output, config, excluded_sfens)


def _prepare(
    root: Path,
    output: Path,
    config: dict,
    excluded_sfens: list[str],
    *,
    extra_excluded_keys: set[str] | None = None,
    dataset_name: str = "dataset",
) -> dict:
    """Two passes remove ALL cross-split duplicate/symmetry states, then encode unique rows."""
    if not (output / "generation-complete.json").exists():
        raise ValueError("generation has not completed")
    if (output / "generation.json").read_bytes() != encoded(config):
        raise ValueError("preparation generation config changed")
    start = config.get("first_game", 0)
    expected_games = set(range(start, start + config["games"]))
    if config.get("recovery_policy"):
        expected_games = {
            json.loads(p.read_text())["game"] for p in (output / "games").glob("*.receipt.json")
        }
    actual_paths = {p.name for p in (output / "games").glob("*.json.gz")}
    if actual_paths != {f"{game:06d}.json.gz" for game in expected_games}:
        raise ValueError("source trajectory set incomplete or contains unexpected games")
    guard_path = root / config["split_guard_path"]
    if digest(guard_path) != config["split_guard_sha256"]:
        raise ValueError("split guard changed")
    exclusions_sha256 = hashlib.sha256(encoded(sorted(excluded_sfens))).hexdigest()
    development_keys = set()
    if config.get("development_exclusions_path"):
        path = root / config["development_exclusions_path"]
        if digest(path) != config["development_exclusions_sha256"]:
            raise ValueError("development exclusions changed")
        development_keys = set(json.loads(path.read_text())["symmetry_keys"])
    dataset = output / dataset_name
    if (dataset / "manifest.json").exists():
        report = json.loads((dataset / "manifest.json").read_text())
        if (
            report.get("split_guard_sha256") != digest(guard_path)
            or report.get("generation_sha256") != digest(output / "generation.json")
            or report.get("exclusions_sha256") != exclusions_sha256
        ):
            raise ValueError("prepared dataset configuration/exclusions changed")
        for ref in report["artifacts"]:
            if digest(dataset / ref["path"]) != ref["sha256"]:
                raise ValueError("prepared dataset corrupted")
        for ref in report["source_games"]:
            if digest(output / "games" / ref["path"]) != ref["sha256"]:
                raise ValueError("prepared source trajectory corrupted")
        return report
    guard = json.loads(guard_path.read_text())
    excluded = {
        r["position_sha256"]
        for rows in guard.values()
        if isinstance(rows, list)
        for r in rows
        if isinstance(r, dict) and "position_sha256" in r
    }
    for sfen in excluded_sfens:
        excluded.update(symmetry_keys(sfen))
    excluded.update(development_keys)
    excluded.update(extra_excluded_keys or set())
    owners, conflicts, raw_count, raw_mates = {}, set(), 0, 0
    raw_terminals = {"win": 0, "resign": 0}
    games = []
    for path in sorted((output / "games").glob("*.json.gz")):
        receipt = json.loads(path.with_suffix(".receipt.json").read_text())
        if digest(path) != receipt["sha256"]:
            raise ValueError("raw game identity mismatch")
        game = json.loads(gzip.decompress(path.read_bytes()))
        expected_split = partition(game["game"], config["seed"])
        if config.get("defense_campaign"):
            from .defense_scenarios import assignment, rotate_move, rotate_sfen

            family, variant = assignment(config, game["game"])
            expected_split = family["split"]
            expected_prefix = [rotate_move(m) if variant % 2 else m for m in family["moves"]]
            if (
                game.get("family") != family["id"]
                or game.get("variant") != variant
                or game.get("group") != family["group"]
                or game.get("prefix_moves") != expected_prefix
                or game.get("initial_sfen") != (rotate_sfen(START) if variant % 2 else START)
            ):
                raise ValueError("source family lineage changed")
        if (
            game["game"] not in expected_games
            or game["seed"] != config["seed"] + game["game"]
            or game["split"] != expected_split
            or path.name != f"{game['game']:06d}.json.gz"
        ):
            raise ValueError("source trajectory split changed")
        games.append(
            {
                "path": path.name,
                "sha256": receipt["sha256"],
                "game": game["game"],
                "split": game["split"],
            }
        )
        if config.get("defense_campaign"):
            games[-1].update(family=game["family"], group=game["group"])
        for sfen, score, *_ in _observations(game):
            raw_count += 1
            if score["kind"] == "terminal":
                outcome = score.get("outcome")
                if outcome not in raw_terminals or score.get("validation") != (
                    "native_csa_28_27" if outcome == "win" else "native_no_legal_moves"
                ):
                    raise ValueError("unvalidated typed teacher terminal observation")
                raw_terminals[outcome] += 1
                continue
            if score["kind"] == "mate":
                raw_mates += 1
                continue
            if score["kind"] != "cp":
                raise ValueError("unknown teacher score kind")
            keys = symmetry_keys(sfen)
            key = min(keys)
            if keys & excluded:
                conflicts.add(key)
            if key in owners and owners[key] != game["split"]:
                conflicts.add(key)
            owners[key] = game["split"]
    if {g["game"] for g in games} != expected_games:
        raise ValueError("source trajectory set incomplete")
    source_cap_removed = 0
    if config.get("defense_campaign"):
        cap = config["defense_campaign"]["source_row_caps"]["generated"]
        eligible = sorted(
            key for key, owner in owners.items() if owner == "train" and key not in conflicts
        )
        rejected = eligible[cap:]
        source_cap_removed = len(rejected)
        conflicts.update(rejected)
    dataset.mkdir(exist_ok=True)
    counts = {
        s: sum(owner == s and key not in conflicts for key, owner in owners.items())
        for s in ("train", "validation", "development_test")
    }
    arrays, streams = {}, {}
    with ExitStack() as stack:
        for split, count in counts.items():
            arrays[split] = {
                "features": np.lib.format.open_memmap(
                    dataset / f"{split}-features.npy",
                    mode="w+",
                    dtype="uint16",
                    shape=(count, 2, MAX_FEATURES),
                ),
                "lengths": np.lib.format.open_memmap(
                    dataset / f"{split}-lengths.npy", mode="w+", dtype="uint8", shape=(count, 2)
                ),
                "targets": np.lib.format.open_memmap(
                    dataset / f"{split}-targets.npy", mode="w+", dtype="float32", shape=(count,)
                ),
            }
            if config.get("defense_campaign"):
                arrays[split]["groups"] = np.lib.format.open_memmap(
                    dataset / f"{split}-groups.npy", mode="w+", dtype="uint8", shape=(count,)
                )
            streams[split] = stack.enter_context(
                gzip.open(dataset / f"{split}-rows.jsonl.gz", "wb")
            )
        used, offsets, distributions = set(), dict.fromkeys(counts, 0), {s: {} for s in counts}
        for ref in games:
            game = json.loads(gzip.decompress((output / "games" / ref["path"]).read_bytes()))
            split = game["split"]
            for sfen, score, kind, ply, index, move in _observations(game):
                if score["kind"] != "cp":
                    continue
                key = min(symmetry_keys(sfen))
                if key in used or key in conflicts:
                    continue
                value = score["value"]
                if type(value) is not int or abs(value) > 28999:
                    raise ValueError("teacher cp outside observed namespace")
                used.add(key)
                black, white, stm = sparse_features(sfen)
                own, other = (white, black) if stm else (black, white)
                if max(len(own), len(other)) > MAX_FEATURES:
                    raise ValueError("sparse feature capacity exceeded")
                n = offsets[split]
                for side, features in enumerate((own, other)):
                    arrays[split]["features"][n, side, : len(features)] = features
                    arrays[split]["lengths"][n, side] = len(features)
                arrays[split]["targets"][n] = np.clip(value, -20000, 20000)
                row = {
                    "sfen": sfen,
                    "score": score,
                    "score_perspective": "side_to_move",
                    "kind": kind,
                    "game": game["game"],
                    "raw_record": index,
                    "move": move,
                    "ply": ply,
                    "position_sha256": position_hash(sfen),
                    "symmetry_key": key,
                }
                if config.get("defense_campaign"):
                    from .defense_scenarios import GROUPS, row_group

                    group = row_group(game, ply)
                    row.update(
                        group=group,
                        family=game["family"],
                        source="generated",
                        teacher_requested_root_depth=(
                            config["teacher_depth"]
                            if kind in ("root", "candidate")
                            else config["defense_campaign"]["relabel_depth"]
                        ),
                        teacher_label_origin=(
                            "root_multipv_child_negated"
                            if kind == "candidate"
                            else "direct_root"
                            if kind == "root"
                            else "direct_focus"
                        ),
                    )
                    arrays[split]["groups"][n] = GROUPS[group]
                    distributions[split][group] = distributions[split].get(group, 0) + 1
                streams[split].write(encoded(row) + b"\n")
                offsets[split] += 1
                stage = "opening" if ply < 40 else "middle" if ply < 110 else "end"
                stage_label = f"stage_{stage}" if config.get("defense_campaign") else stage
                for label in (
                    stage_label,
                    kind,
                    "saturated" if abs(value) > 20000 else "unsaturated",
                ):
                    distributions[split][label] = distributions[split].get(label, 0) + 1
    for group in arrays.values():
        for array in group.values():
            array.flush()
    if offsets != counts:
        raise ValueError("unique dataset count mismatch")
    label_summary = {}
    for split, group in arrays.items():
        targets = group["targets"]
        label_summary[split] = {
            "count": len(targets),
            "distinct_cp": len(np.unique(targets)),
            "minimum": float(targets.min()) if len(targets) else None,
            "maximum": float(targets.max()) if len(targets) else None,
            "standard_deviation": float(targets.std()) if len(targets) else None,
            "zero_labels": int((targets == 0).sum()),
        }
    artifacts = [
        {"path": p.name, "sha256": digest(p), "bytes": p.stat().st_size}
        for p in sorted(dataset.iterdir())
        if p.is_file()
    ]
    report = {
        "schema": SCHEMA,
        "unique_positions": counts,
        "raw_observations": raw_count,
        "source_cap_removed": source_cap_removed,
        "mate_observations_masked": raw_mates,
        "teacher_terminal_observations_masked": raw_terminals,
        "excluded_conflict_keys": len(conflicts),
        "duplicate_or_excluded_observations": raw_count
        - raw_mates
        - sum(raw_terminals.values())
        - sum(counts.values()),
        "source_games": games,
        "distributions": distributions,
        "label_summary": label_summary,
        "artifacts": artifacts,
        "split_guard_sha256": digest(guard_path),
        "generation_sha256": digest(output / "generation.json"),
        "exclusions_sha256": exclusions_sha256,
        "sealed_holdout_opened": False,
        "excluded_preflight_symmetry_keys": len(development_keys),
        "symmetry_cross_split_overlap": 0,
        "independence_limit": (
            "Own trajectories share the initial position and teacher; "
            "no claim of independent human games."
        ),
    }
    atomic(dataset / "manifest.json", encoded(report))
    return report
