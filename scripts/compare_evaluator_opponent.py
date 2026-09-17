"""Two book-free, fixed-50ms supplementary games; no rank or clock-match claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from pathlib import Path

from open_shogi_training.evaluator_arena import _inside, _reference
from open_shogi_training.evaluator_data import START, Replay, atomic, digest, encoded
from open_shogi_training.phase10u_arena_evidence import PROHIBITED, proof
from open_shogi_training.phase10u_arena_runner import Player


def ending(terminal: str) -> tuple[str, str | None]:
    if terminal == "None":
        return "ongoing", None
    if terminal == "Some(Repetition(NoContest))":
        return "repetition", None
    match = re.fullmatch(r"Some\(Checkmate \{ winner: (Black|White) \}\)", terminal)
    if match:
        return "checkmate", match[1].lower()
    for pattern, reason in (
        (r"Some\(NoLegalMoves \{ loser: (Black|White) \}\)", "no_legal_moves"),
        (r"Some\(Repetition\(PerpetualCheckLoss\((Black|White)\)\)\)", "perpetual_check"),
    ):
        match = re.fullmatch(pattern, terminal)
        if match:
            return reason, "white" if match[1] == "Black" else "black"
    raise ValueError(f"unknown native termination: {terminal}")


def validate_response(response: dict, state: dict, handcrafted: bool) -> None:
    if response["final_sfen"] != state["sfen"] or response["termination"] == "EvaluationError":
        raise ValueError("wrong searched state or failed evaluation")
    proof(
        {key: response["proof"][key] for key in (*PROHIBITED, "learned_eval_calls")},
        handcrafted=handcrafted,
    )
    if not response["deadline"]["hard_compliant"]:
        raise ValueError("player exceeded hard deadline")
    if type(response["score"]) is not int or response["best_move"] not in {
        child["move"] for child in state["successors"]
    }:
        raise ValueError("nonfinite/missing score or illegal proposed move")


def replay_config(plan: dict) -> dict:
    return {
        "replay_path": plan["replay"]["path"],
        "leaf_path": plan["candidate"]["model"]["path"],
        "leaf_sha256": plan["candidate"]["model"]["sha256"],
    }


def verified_result(root: Path, path: Path, plan: dict, plan_sha: str) -> dict:
    result = json.loads(path.read_text())
    checksum = result.pop("sha256")
    if hashlib.sha256(encoded(result)).hexdigest() != checksum or result["plan_sha256"] != plan_sha:
        raise ValueError("supplementary receipt changed")
    for ref in result["artifacts"]:
        _reference(root, ref["path"], ref["sha256"])
    replay = Replay(root, replay_config(plan))
    try:
        state = replay.ask(reset=START)
        for movement in result["moves"]:
            state = replay.ask(movement=movement)
        if state["sfen"] != result["final_sfen"] or ending(state["terminal"]) != (
            result["native_reason"],
            result["winner"],
        ):
            raise ValueError("saved game does not reproduce")
    finally:
        replay.close()
    return {**result, "sha256": checksum}


def play(root: Path, folder: Path, plan: dict, side: str) -> dict:
    folder.mkdir()
    players = {}
    replay = Replay(root, replay_config(plan))
    moves, events = [], []
    started = time.monotonic()
    try:
        for role in ("candidate", "opponent"):
            players[role] = Player(root, plan[role], folder / f"{role}.jsonl")
        state = replay.ask(reset=START, successors=True)
        for _ in range(256):
            if ending(state["terminal"])[0] != "ongoing":
                break
            active = "black" if state["sfen"].split()[1] == "b" else "white"
            role = "candidate" if active == side else "opponent"
            request = {
                "initial_sfen": START,
                "moves": moves.copy(),
                "depth": 64,
                "nodes": None,
                "movetime_ms": 50,
                "hard_timeout_ms": 2000,
                "hash_mb": 32,
            }
            response, timing = players[role].search(request)
            validate_response(response, state, role == "opponent")
            movement = response["best_move"]
            state = replay.ask(movement=movement, successors=True)
            moves.append(movement)
            events.append(
                {
                    "side": active,
                    "role": role,
                    "request": request,
                    "response": response,
                    "timing": timing,
                    "result_sfen": state["sfen"],
                }
            )
            atomic(folder / "progress.json", encoded({"plies": len(moves), "candidate_side": side}))
        reason, winner = ending(state["terminal"])
        trace = folder / "trace.json"
        atomic(trace, encoded(events))
        result = {
            "schema": "open_shogiai_supplementary_opponent/v1",
            "candidate_side": side,
            "plan_sha256": hashlib.sha256(encoded(plan)).hexdigest(),
            "status": "completed" if reason != "ongoing" else "incomplete",
            "reason": reason if reason != "ongoing" else "max_plies_unscored",
            "native_reason": reason,
            "winner": winner,
            "moves": moves,
            "final_sfen": state["sfen"],
            "wall_seconds": time.monotonic() - started,
            "score_candidate": (0.5 if winner is None else float(winner == side))
            if reason != "ongoing"
            else None,
            "soft_deadline_overruns": sum(
                not e["response"]["deadline"]["soft_compliant"] for e in events
            ),
            "hard_deadline_violations": 0,
            "all_moves_native_legal": True,
            "artifacts": [
                _reference(root, p)
                for p in (trace, folder / "candidate.jsonl", folder / "opponent.jsonl")
            ],
        }
        result["sha256"] = hashlib.sha256(encoded(result)).hexdigest()
        atomic(folder / "result.json", encoded(result))
        return result
    except Exception as error:
        atomic(
            folder / "failure.json",
            encoded({"error": f"{type(error).__name__}: {error}", "moves": moves}),
        )
        raise
    finally:
        replay.close()
        for player in players.values():
            player.close()


def run(root: Path, run_path: Path, output: Path) -> dict:
    output = _inside(root, output, file=False)
    if not output.is_relative_to(root / "local/post-training-evidence"):
        raise ValueError("supplementary output must remain in local/post-training-evidence")
    output.mkdir(parents=True, exist_ok=True)
    run_path = _inside(root, run_path, file=False)
    training = json.loads((run_path / "fit/training.json").read_text())
    candidate = _reference(root, run_path / "fit/best.osaval03", training["best_sha256"])
    plan = {
        "schema": "open_shogiai_supplementary_opponent/v1",
        "driver": _reference(root, Path(__file__).resolve()),
        "candidate_step": training["best_step"],
        "candidate": {},
        "opponent": {},
        "replay": _reference(root, run_path / "runtime/position_audit"),
        "games": ["black", "white"],
        "movetime_ms": 50,
        "max_plies": 256,
        "classification": (
            "regression control; shared search/rules; separate handcrafted evaluation"
        ),
        "limits": (
            "existing SHA-pinned binaries; current-HEAD build provenance unverified; "
            "no shodan calibration; not 3/10-minute games; paired games are related"
        ),
        "book": "off",
        "promotion": False,
    }
    for role, source in (
        ("candidate", "target/pure/release/open-shogi-cli"),
        ("opponent", "target/release/open-shogi-cli"),
    ):
        ref = _reference(root, source)
        native = output / "artifacts" / ref["sha256"] / "open-shogi-cli"
        native.parent.mkdir(parents=True, exist_ok=True)
        if not native.exists():
            shutil.copy2(root / source, native)
        plan[role] = {
            "native": _reference(root, native, ref["sha256"]),
            "profile_name": "pure_learned" if role == "candidate" else "handcrafted_experimental",
            "model": candidate if role == "candidate" else None,
            "adapter_format": "OSAVAL03" if role == "candidate" else "HANDCRAFTED",
        }
    path = output / "plan.json"
    if path.exists() and path.read_bytes() != encoded(plan):
        raise ValueError("supplementary plan changed; preserve previous evidence")
    atomic(path, encoded(plan))
    results = []
    for side in plan["games"]:
        folder = output / side
        if (folder / "result.json").exists():
            results.append(verified_result(root, folder / "result.json", plan, digest(path)))
        elif folder.exists():
            raise ValueError(
                "interrupted game retained; do not duplicate or reset its finite attempt"
            )
        else:
            results.append(play(root, folder, plan, side))
    report = {
        "plan_sha256": digest(path),
        "games": results,
        "planned_games": 2,
        "completed_games": sum(r["status"] == "completed" for r in results),
        "human_shodan_validated": False,
        "main_arena_games_added": 0,
    }
    atomic(output / "result.json", encoded(report))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(Path(__file__).resolve().parents[1], args.run, args.output)
    print(
        json.dumps(
            {
                "planned_games": 2,
                "completed_games": result["completed_games"],
                "scores": [game["score_candidate"] for game in result["games"]],
            }
        )
    )
