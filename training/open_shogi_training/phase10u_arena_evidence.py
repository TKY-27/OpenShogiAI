"""Fail-closed immutable pure Arena receipts; this module never launches an Arena.

Trust is supplied by an independently reviewed manifest and its out-of-band digest.
A self-hash detects corruption, not authenticity: receipts cannot certify themselves.
The rules-only oracle reconstructs the raw game independently of the playing processes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

SCHEMA = "open_shogiai_phase10u_arena_receipt/v1"
PROHIBITED = (
    "handcrafted_eval_calls",
    "residual_eval_calls",
    "composite_eval_calls",
    "fallback_count",
    "book_hits",
    "teacher_calls",
)
WARMUP = "none; model startup excluded for both sides; all request setup charged"


class EvidenceError(ValueError):
    """Missing, changed or unverifiable evidence keeps execution closed."""


def canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def read_receipt(path: Path) -> dict:
    """Reject duplicate keys, alternate serialization and noncanonical receipt bytes."""
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or canonical(value) != raw or seal(value) != value:
        raise EvidenceError("receipt bytes are not canonical or digest is invalid")
    return value


def file_sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise EvidenceError(f"not a regular artifact: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(root: Path, ref: dict) -> bytes:
    path = Path(ref["path"])
    if path.is_absolute() or ".." in path.parts:
        raise EvidenceError("artifact must be repository-relative")
    target = root / path
    if not target.resolve().is_relative_to(root.resolve()) or file_sha(target) != ref["sha256"]:
        raise EvidenceError("immutable artifact SHA-256 mismatch")
    return target.read_bytes()


def proof(value: dict, *, handcrafted: bool = False) -> None:
    if set(value) != {*PROHIBITED, "learned_eval_calls"}:
        raise EvidenceError("runtime counters missing or unknown")
    if any(type(v) is not int or v < 0 for v in value.values()):
        raise EvidenceError("invalid runtime counters")
    positive = "handcrafted_eval_calls" if handcrafted else "learned_eval_calls"
    if value[positive] <= 0 or any(v != 0 for k, v in value.items() if k != positive):
        raise EvidenceError("prohibited runtime or missing required evaluation")


def validate_identity(root: Path, identity: dict) -> None:
    """Validate every side's separately reviewed build and full immutable data binding."""
    handcrafted = identity["adapter_format"] == "HANDCRAFTED"
    features = ["handcrafted"] if handcrafted else ["pure-only"]
    if identity["profile_name"] != ("handcrafted_experimental" if handcrafted else "pure_learned"):
        raise EvidenceError("wrong runtime profile")
    if identity["cargo_features"] != features:
        raise EvidenceError("wrong Cargo feature set")
    if len(identity["source_commit"]) != 40 or len(identity["evaluator_schema_sha256"]) != 64:
        raise EvidenceError("missing source or evaluator schema hash")
    audit = json.loads(artifact(root, identity["build_audit"]))
    if audit["status"] != "PASS":
        raise EvidenceError("uncertified build")
    for key in (
        "source_commit",
        "cargo_features",
        "adapter_format",
        "native",
        "wasm",
        "evaluator_schema_sha256",
    ):
        if canonical(audit[key]) != canonical(identity[key]):
            raise EvidenceError(f"build audit identity mismatch: {key}")
    if not handcrafted and audit["prohibited_modules_absent"] is not True:
        raise EvidenceError("pure binary contains prohibited modules")
    artifact(root, identity["native"])
    profile_bytes = artifact(root, identity["profile"])
    if identity["evaluator_schema_sha256"] != identity["profile"]["sha256"]:
        raise EvidenceError("evaluator schema must bind exact profile bytes")
    profile = json.loads(profile_bytes)
    if not handcrafted and (
        profile["modelFormat"] != identity["adapter_format"] or profile["profile"] != "pure_learned"
    ):
        raise EvidenceError("profile/model format mismatch")
    if identity["wasm"] is not None:
        artifact(root, identity["wasm"])
    if handcrafted:
        if identity["model"] is not None or identity["wasm"] is not None:
            raise EvidenceError("handcrafted comparison must not load a model")
        return
    data = artifact(root, identity["model"])
    if identity["adapter_format"] == "OSAVAL02":
        from open_shogi_training.phase10r_model import parse_osaval02

        payload = parse_osaval02(data).weight_payload_sha256
    elif identity["adapter_format"] == "OSAT10A1":
        from open_shogi_training.phase10t_model import Phase10TModel

        payload = hashlib.sha256(Phase10TModel.from_bytes(data).payload()).hexdigest()
    elif identity["adapter_format"] == "OSAVAL03":
        from open_shogi_training.phase10v_model import Phase10VModel

        # Validate the closed successor format before binding its exact payload.
        # No Phase 10U receipt or legacy model acquires different semantics.
        Phase10VModel.from_bytes(data)
        payload = hashlib.sha256(data[44:-32]).hexdigest()
    else:
        raise EvidenceError("unknown model format; fallback forbidden")
    if payload != identity["model"]["payload_sha256"]:
        raise EvidenceError("model payload binding changed")


def validate_manifest(root: Path, trusted: dict, trusted_sha256: str) -> None:
    """Preflight every scheduled build/data/control before any player is launched."""
    if digest(trusted) != trusted_sha256:
        raise EvidenceError("reviewed manifest digest mismatch")
    artifact(root, trusted["replay_oracle"])
    starts = json.loads(artifact(root, trusted["start_manifest"]))
    if isinstance(starts, dict):
        starts = starts["starts"]
    allowed = {row["sfen"] for row in starts}
    if not trusted["games"]:
        raise EvidenceError("empty schedule")
    seen = set()
    for game_id, game in trusted["games"].items():
        if not game_id or game["initial_sfen"] not in allowed:
            raise EvidenceError("invalid scheduled start or game identity")
        validate_controls(game["controls"])
        if type(game["seed"]) is not int or game["seed"] < 0 or not game["pairing_id"]:
            raise EvidenceError("invalid pairing or seed")
        for identity in game["sides"].values():
            binding = digest(identity)
            if binding not in seen:
                validate_identity(root, identity)
                seen.add(binding)


def validate_controls(controls: dict) -> None:
    if controls["warmup"] != WARMUP:
        raise EvidenceError("unsupported warmup accounting")
    if controls["clock"] != "monotonic_ns" or controls["threads"] != 1:
        raise EvidenceError("unsupported clock or threads")
    if any(
        type(controls[k]) is not int or controls[k] <= 0
        for k in ("threads", "hash_mb", "max_plies", "hard_timeout_ns", "max_depth")
    ) or not isinstance(controls["search_options"], dict):
        raise EvidenceError("missing or invalid search controls")
    if controls["max_plies"] > 10000 or controls["max_depth"] > 64:
        raise EvidenceError("search bound exceeds supported range")
    expected_options = {
        "configuration": "SearchConfig::default",
        "hash_mb": controls["hash_mb"],
        "fresh_engine_per_move": True,
        "book": False,
    }
    if canonical(controls["search_options"]) != canonical(expected_options):
        raise EvidenceError("unsupported search options")
    soft, nodes = controls["movetime_ns"], controls["nodes"]
    if soft is not None and (type(soft) is not int or soft <= 0 or soft % 1_000_000):
        raise EvidenceError("invalid soft budget")
    if nodes is not None and (type(nodes) is not int or nodes <= 0):
        raise EvidenceError("invalid node budget")
    if (nodes is None) == (soft is None) or controls["hard_timeout_ns"] % 1_000_000:
        raise EvidenceError("exactly one equal clock/node control required")
    if controls["hash_mb"] != 32 or controls["hard_timeout_ns"] > 60_000_000_000:
        raise EvidenceError("controls exceed playing protocol bounds")
    if soft is not None and soft > controls["hard_timeout_ns"]:
        raise EvidenceError("soft budget exceeds hard deadline")


def seal(receipt: dict) -> dict:
    value = json.loads(canonical(receipt))
    value.pop("receipt_sha256", None)
    value["replay_sha256"] = digest(
        {
            key: value[key]
            for key in (
                "initial_sfen",
                "moves",
                "result",
                "termination",
                "max_plies_reached",
                "final_sfen",
            )
        }
    )
    return {**value, "receipt_sha256": digest(value)}


def write_immutable(path: Path, receipt: dict) -> None:
    with path.open("xb") as output:
        output.write(canonical(seal(receipt)))
        output.flush()
        import os

        os.fsync(output.fileno())


class MonotonicSearch:
    """Capture per-search wall deadline in the coordinator's monotonic clock domain."""

    def __init__(self, budget_ns: int):
        if type(budget_ns) is not int or budget_ns <= 0:
            raise EvidenceError("invalid search budget")
        self.start_ns = time.monotonic_ns()
        self.deadline_ns = self.start_ns + budget_ns

    def finish(self) -> dict:
        end = time.monotonic_ns()
        return {
            "clock": "monotonic_ns",
            "start_ns": self.start_ns,
            "deadline_ns": self.deadline_ns,
            "end_ns": end,
            "elapsed_ns": end - self.start_ns,
            "compliant": end <= self.deadline_ns,
        }


def validate(root: Path, receipt: dict, trusted: dict, trusted_sha256: str) -> dict:
    """Verify evidence against an independently pinned manifest, never receipt-supplied trust."""
    try:
        return _validate(root, receipt, trusted, trusted_sha256)
    except (KeyError, TypeError, OSError, json.JSONDecodeError) as error:
        raise EvidenceError(f"incomplete or invalid evidence: {error}") from error


def _validate(root: Path, receipt: dict, trusted: dict, trusted_sha256: str) -> dict:
    if digest(trusted) != trusted_sha256 or receipt["manifest_sha256"] != trusted_sha256:
        raise EvidenceError("trusted manifest binding changed")
    if receipt["schema"] != SCHEMA or seal(receipt) != receipt:
        raise EvidenceError("canonical receipt or replay digest mismatch")
    game = trusted["games"][receipt["game_id"]]
    for key in ("initial_sfen", "seed", "pairing_id", "sides", "controls"):
        if canonical(receipt[key]) != canonical(game[key]):
            raise EvidenceError(f"immutable game binding changed: {key}")
    artifact(root, trusted["start_manifest"])
    if receipt["start_manifest_sha256"] != trusted["start_manifest"]["sha256"]:
        raise EvidenceError("start manifest mismatch")
    starts = json.loads(artifact(root, trusted["start_manifest"]))
    if isinstance(starts, dict):
        starts = starts["starts"]
    if receipt["initial_sfen"] not in [row["sfen"] for row in starts]:
        raise EvidenceError("start is outside frozen manifest")
    for side in ("black", "white"):
        identity = receipt["sides"][side]
        validate_identity(root, identity)
        proof(
            receipt["runtime"][side],
            handcrafted=identity["adapter_format"] == "HANDCRAFTED",
        )
    controls = receipt["controls"]
    validate_controls(controls)
    moves, searches = receipt["moves"], receipt["searches"]
    if len(searches) != len(moves) or not searches:
        raise EvidenceError("every move requires per-search evidence")
    previous_end = -1
    first_side = "black" if receipt["initial_sfen"].split()[1] == "b" else "white"
    for index, search in enumerate(searches):
        timing = search["timing"]
        side = first_side if index % 2 == 0 else ("white" if first_side == "black" else "black")
        if search["side"] != side or search["move"] != moves[index]:
            raise EvidenceError("search sequence mismatch")
        if any(
            type(timing[k]) is not int for k in ("start_ns", "end_ns", "elapsed_ns", "deadline_ns")
        ):
            raise EvidenceError("deadline evidence must be integer nanoseconds")
        if (
            timing["clock"] != "monotonic_ns"
            or timing["start_ns"] < 0
            or timing["start_ns"] < previous_end
            or timing["end_ns"] < timing["start_ns"]
            or timing["deadline_ns"] != timing["start_ns"] + controls["hard_timeout_ns"]
            or timing["elapsed_ns"] != timing["end_ns"] - timing["start_ns"]
            or timing["end_ns"] > timing["deadline_ns"]
            or timing["compliant"] is not True
        ):
            raise EvidenceError("per-search deadline violated or evidence inconsistent")
        if "request" not in search or "response" not in search:
            raise EvidenceError("missing complete raw player exchange")
        if "request" in search:
            request, response = search["request"], search["response"]
            expected = {
                "initial_sfen": receipt["initial_sfen"],
                "moves": moves[:index],
                "depth": controls["max_depth"],
                "nodes": controls["nodes"],
                "movetime_ms": None
                if controls["movetime_ns"] is None
                else controls["movetime_ns"] // 1_000_000,
                "hard_timeout_ms": controls["hard_timeout_ns"] // 1_000_000,
                "hash_mb": controls["hash_mb"],
            }
            identity = receipt["sides"][side]
            if (
                canonical(request) != canonical(expected)
                or canonical(response["requested_controls"]) != canonical(expected)
                or canonical(response["timing"]) != canonical(search["engine_timing"])
                or response["best_move"] != search["move"]
                or response["model_format"] != identity["adapter_format"]
                or response["model_sha256"]
                != (identity["model"]["sha256"] if identity["model"] else "")
                or (
                    identity["model"] is not None
                    and (
                        response["proof"]["model_sha256"] != identity["model"]["sha256"]
                        or response["proof"]["evaluator_profile_schema_hash"]
                        != identity["evaluator_schema_sha256"]
                    )
                )
                or response["threads"] != controls["threads"]
                or response["hash_mb"] != controls["hash_mb"]
                or response["depth_limit"] != controls["max_depth"]
                or response["node_limit"] != controls["nodes"]
                or canonical({k: response["proof"][k] for k in search["runtime"]})
                != canonical(search["runtime"])
            ):
                raise EvidenceError("raw player request/response binding mismatch")
        engine = search["engine_timing"]
        if engine["clock"] != "monotonic" or engine["scope"] != "request_parse_replay_setup_search":
            raise EvidenceError("wrong engine timing clock or scope")
        for key in (
            "setup_elapsed_ns",
            "search_start_ns",
            "search_budget_ns",
            "search_elapsed_ns",
            "elapsed_ns",
            "hard_budget_ns",
            "hard_timeout_ms",
        ):
            if type(engine[key]) is not int or engine[key] < 0:
                raise EvidenceError("invalid engine timing")
        soft = controls["movetime_ns"]
        hard = controls["hard_timeout_ns"]
        soft_setup_overrun = engine.get("soft_budget_exhausted_in_setup", False)
        hard_deadline_safety_margin = engine.get("hard_deadline_safety_margin_ns", 0)
        if type(soft_setup_overrun) is not bool:
            raise EvidenceError("invalid soft setup-overrun evidence")
        if type(hard_deadline_safety_margin) is not int or hard_deadline_safety_margin < 0:
            raise EvidenceError("invalid hard deadline safety margin")
        if soft is not None and engine["setup_elapsed_ns"] >= soft:
            expected_search_budget = max(
                0, hard - engine["setup_elapsed_ns"] - hard_deadline_safety_margin
            )
            if soft_setup_overrun is not True:
                raise EvidenceError("missing soft setup-overrun evidence")
            if hard_deadline_safety_margin != 5_000_000:
                raise EvidenceError("invalid hard deadline safety margin")
        else:
            expected_search_budget = max(
                0, (soft if soft is not None else hard) - engine["setup_elapsed_ns"]
            )
            if soft_setup_overrun is not False:
                raise EvidenceError("unexpected soft setup-overrun evidence")
            if hard_deadline_safety_margin != 0:
                raise EvidenceError("unexpected hard deadline safety margin")
        if (
            engine["soft_budget_ns"] != soft
            or engine["hard_budget_ns"] != hard
            or engine["requested_movetime_ms"] != (None if soft is None else soft // 1_000_000)
            or engine["hard_timeout_ms"] * 1_000_000 != hard
            or engine["search_start_ns"] < engine["setup_elapsed_ns"]
            or engine["search_budget_ns"] != expected_search_budget
            or engine["elapsed_ns"] < engine["search_start_ns"] + engine["search_elapsed_ns"]
            or engine["elapsed_ns"] > timing["elapsed_ns"]
            or engine["elapsed_ns"] > hard
            or engine["hard_compliant"] is not True
            or engine["compliant"] is not True
            or engine["soft_compliant"]
            is not (None if soft is None else engine["elapsed_ns"] <= soft)
        ):
            raise EvidenceError("engine deadline evidence mismatch")
        proof(
            search["runtime"],
            handcrafted=receipt["sides"][side]["adapter_format"] == "HANDCRAFTED",
        )
        previous_end = timing["end_ns"]
    for side in ("black", "white"):
        for key in (*PROHIBITED, "learned_eval_calls"):
            if receipt["runtime"][side][key] != sum(
                s["runtime"][key] for s in searches if s["side"] == side
            ):
                raise EvidenceError("runtime aggregate does not match searches")
    artifact(root, trusted["replay_oracle"])
    oracle = root / trusted["replay_oracle"]["path"]
    request = {
        "initial_sfen": receipt["initial_sfen"],
        "moves": moves,
        "max_plies": controls["max_plies"],
    }
    run = subprocess.run([str(oracle)], input=canonical(request), capture_output=True, timeout=30)
    if run.returncode:
        raise EvidenceError("independent raw replay rejected")
    replay = json.loads(run.stdout)
    for index, search in enumerate(searches):
        prefix = {**request, "moves": moves[:index]}
        checked = subprocess.run(
            [str(oracle)], input=canonical(prefix), capture_output=True, timeout=30
        )
        if (
            checked.returncode
            or search["response"]["final_sfen"] != json.loads(checked.stdout)["final_sfen"]
        ):
            raise EvidenceError("player searched wrong replay position")
    for key in ("final_sfen", "termination", "result", "max_plies_reached"):
        if canonical(receipt[key]) != canonical(replay[key]):
            raise EvidenceError(f"replay classification mismatch: {key}")
    if replay["termination"] == "ongoing":
        raise EvidenceError("incomplete game")
    # Recheck all executable/data bytes after replay to detect mutation during verification.
    for side in receipt["sides"].values():
        for key in ("native", "model", "profile", "wasm", "build_audit"):
            if side[key] is not None:
                artifact(root, side[key])
    artifact(root, trusted["replay_oracle"])
    return replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--receipt", type=Path, action="append", default=[])
    args = parser.parse_args()
    trusted = json.loads(args.manifest.read_bytes())
    root = args.root.resolve()
    validate_manifest(root, trusted, args.manifest_sha256)
    for path in args.receipt:
        validate(root, read_receipt(path), trusted, args.manifest_sha256)
    print(
        json.dumps(
            {
                "status": "verified",
                "planned_games": len(trusted["games"]),
                "receipts_replayed": len(args.receipt),
                "execution_authorized": False,
            }
        )
    )


if __name__ == "__main__":
    main()
