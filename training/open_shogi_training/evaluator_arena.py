"""Finite, paired, same-runtime evaluator comparison; never promotes a model.

The caller runs this only after training, with no competing training/teacher jobs.
Only development-test metadata supplies starts; sealed holdout files are not opened.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .evaluator_data import START, atomic, digest, encoded, symmetry_keys

SCHEMA = "open_shogiai_evaluator_arena/v1"
PROFILE_HASH = "d2eec27887926ccc8a076552815cd54e34b85d6d23e65732ddba4989bf59c1e7"
FORBIDDEN = (
    "handcrafted_eval_calls",
    "residual_eval_calls",
    "composite_eval_calls",
    "book_hits",
    "teacher_calls",
    "fallback_count",
)
NS_PER_MS = 1_000_000


def _inside(root: Path, value: str | Path, *, file: bool = True) -> Path:
    requested = Path(value)
    requested = requested if requested.is_absolute() else root / requested
    if not requested.is_relative_to(root) or any(
        p.is_symlink() for p in (requested, *requested.parents)
    ):
        raise ValueError("arena path is outside its repository or uses a symlink")
    result = requested.resolve(strict=file)
    if not result.is_relative_to(root) or (file and not result.is_file()):
        raise ValueError("arena requires an owned regular repository path")
    return result


def _reference(root: Path, value: str | Path, expected: str | None = None) -> dict:
    path = _inside(root, value)
    sha = digest(path)
    if expected is not None and sha != expected:
        raise ValueError(f"arena artifact changed: {path.relative_to(root)}")
    return {"path": str(path.relative_to(root)), "sha256": sha, "bytes": path.stat().st_size}


def _starts(dataset: Path, seed: int) -> tuple[list[dict], dict]:
    manifest = json.loads((dataset / "manifest.json").read_text())
    ref = next(r for r in manifest["artifacts"] if r["path"] == "development_test-rows.jsonl.gz")
    rows_path = dataset / ref["path"]
    if rows_path.is_symlink() or digest(rows_path) != ref["sha256"]:
        raise ValueError("development-test metadata identity changed")
    permitted_games = {
        r["game"] for r in manifest["source_games"] if r["split"] == "development_test"
    }
    by_game: dict[int, list[dict]] = {}
    with gzip.open(rows_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["game"] not in permitted_games:
                raise ValueError("development-test row belongs to another partition")
            if row["kind"] == "root" and 12 <= row["ply"] <= 32:
                by_game.setdefault(row["game"], []).append(row)
    starts, seen = [], set()
    # Hash ordering does not depend on labels or model performance.
    game_order = sorted(by_game, key=lambda g: hashlib.sha256(encoded([seed, g])).hexdigest())
    for game in game_order:
        candidates = sorted(by_game[game], key=lambda r: (r["ply"] != 16, r["ply"], r["sfen"]))
        for row in candidates:
            key = min(symmetry_keys(row["sfen"]))
            if key not in seen:
                starts.append(
                    {"game": game, "ply": row["ply"], "sfen": row["sfen"], "symmetry_key": key}
                )
                seen.add(key)
                break
        if len(starts) == 16:
            return starts, ref
    raise ValueError("need 16 distinct development-test trajectory roots at ply 16 or 12..32")


def _defense_starts(
    dataset: Path, seed: int, groups: list[str], preferred_families: list[str] | None = None
) -> tuple[list[dict], dict]:
    """Four predeclared strata, four distinct games each; never rank by scores."""
    manifest = json.loads((dataset / "manifest.json").read_text())
    ref = next(r for r in manifest["artifacts"] if r["path"] == "development_test-rows.jsonl.gz")
    path = dataset / ref["path"]
    if path.is_symlink() or digest(path) != ref["sha256"]:
        raise ValueError("development-test metadata identity changed")
    permitted = {g["game"] for g in manifest["source_games"] if g["split"] == "development_test"}
    bounds = {
        "general": (32, 128),
        "opening": (12, 40),
        "defense": (16, 96),
        "attack_end": (80, 191),
    }
    pools = {g: [] for g in groups}
    with gzip.open(path, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            group = row["group"]
            if row["game"] not in permitted:
                raise ValueError("development-test lineage mismatch")
            lo, hi = bounds[group]
            if row["family"] in (preferred_families or []):
                lo = 0  # Authored early-contact tests are selected before results are observed.
            if row["kind"] == "root" and lo <= row["ply"] <= hi:
                pools[group].append(row)
    selected, seen = {}, set()
    for group in groups:
        selected[group], games = [], set()
        ordered = sorted(
            pools[group],
            key=lambda r: hashlib.sha256(encoded([seed, r["game"], r["ply"]])).digest(),
        )
        preferred = preferred_families or []
        reserved = [next((r for r in ordered if r["family"] == f), None) for f in preferred]
        ordered = [r for r in reserved if r is not None] + [
            r for r in ordered if r["family"] not in preferred
        ]
        for row in ordered:
            key = min(symmetry_keys(row["sfen"]))
            if row["game"] in games or key in seen:
                continue
            selected[group].append(row)
            games.add(row["game"])
            seen.add(key)
            if len(games) == 4:
                break
        if len(games) != 4:
            raise ValueError(f"need four distinct held-out trajectory roots in {group}")
    return [selected[g][i] for i in range(4) for g in groups], ref


def _plan(root: Path, config: dict) -> dict:
    if (
        config.get("depth") != 64
        or config.get("max_plies") != 256
        or (not config.get("groups") and config.get("seed") != 20260915)
    ):
        raise ValueError("arena depth, ply ceiling and seed must match the reviewed contract")
    if not 1 <= config.get("max_wall_seconds", 86_400) <= 86_400:
        raise ValueError("arena wall ceiling must be 1..86400 seconds")
    assets = {}
    for kind in ("probe", "baseline", "candidate"):
        assets[kind] = _reference(root, config[f"{kind}_path"], config[f"{kind}_sha256"])
    dataset = _inside(root, config["dataset_path"], file=False)
    assets["dataset_manifest"] = _reference(root, dataset / "manifest.json")
    defense = bool(config.get("groups"))
    starts, rows = (
        _defense_starts(dataset, config["seed"], config["groups"], config.get("preferred_families"))
        if defense
        else _starts(dataset, config["seed"])
    )
    assets["development_test_rows"] = _reference(root, dataset / rows["path"], rows["sha256"])
    assets["probe_driver"] = _reference(root, "scripts/compare_core_prototype.py")
    games = []
    for pair, start in enumerate(starts):
        remaining = 180_000 if pair < 12 else 600_000
        for candidate_side in ("black", "white"):
            games.append(
                {
                    "id": f"evaluation-{pair:02d}-{candidate_side}",
                    "pair": pair,
                    "group": "evaluation",
                    "clock_ms": remaining,
                    "candidate_side": candidate_side,
                    "initial_sfen": start["sfen"],
                    "source_game": start["game"],
                    "source_ply": start["ply"],
                    **({"stratum": start["group"], "family": start["family"]} if defense else {}),
                }
            )
    for pair in range(2 if defense else 4):
        for candidate_side in ("black", "white"):
            games.append(
                {
                    "id": f"demonstration-{pair:02d}-{candidate_side}",
                    "pair": pair,
                    "group": "demonstration",
                    "clock_ms": 180_000 if pair < (1 if defense else 2) else 600_000,
                    "candidate_side": candidate_side,
                    "initial_sfen": START,
                }
            )
    return {
        "schema": SCHEMA,
        "config": config,
        "artifacts": assets,
        "games": games,
        "controller": "off for both models",
        "opening_book": "off",
        "search": {"depth": 64, "quiescence_depth": 4, "tt_mib": 2, "threads": 1},
        "clock_scope": (
            "request construction, replay/setup/search/transport, response validation "
            "and native move application"
        ),
        "startup": "model preparation before game clock starts; measured separately",
        "root_history": "same restored SFEN; repetition history starts at that root",
        "criteria": {
            "evaluation_games": 32,
            **(
                {"minimum_stratum_score": config["minimum_stratum_score"]}
                if "minimum_stratum_score" in config
                else {}
            ),
            "demonstration_games": 4 if defense else 8,
            "all_planned_games_must_finish": True,
            "no_adverse_attempts": True,
            "candidate_score_each_clock_strictly_above": 0.5,
            "cluster": "source family (including all variants and color reversals)"
            if defense
            else "color-reversed pair from one distinct source trajectory",
            "bootstrap_resamples": 10_000,
            "bootstrap_seed": config["seed"],
            "one_sided_lower_percentile": 5,
            "overall_lower_strictly_above": 0.5,
        },
        "max_process_failure_retries_per_game": 1,
        "max_wall_seconds": min(int(config.get("max_wall_seconds", 86_400)), 86_400),
        "promotion": "never; final judgment belongs to Astra",
        "sealed_holdout_opened": False,
    }


def _validate_response(response: dict, model_sha: str, *, nodes: int | None = None) -> None:
    if (
        response.get("schema") != "open_shogiai_core_probe/v1"
        or response.get("leaf_sha256") != model_sha
    ):
        raise ValueError("wrong probe schema or evaluator identity")
    proof = response.get("proof", {})
    if not isinstance(proof, dict):
        raise ValueError("pure proof is not an object")
    if (
        proof.get("profile") != "pure_learned"
        or proof.get("model_sha256") != model_sha
        or proof.get("profile_schema") != "open_shogiai_pure_learned_v3_profile/v1"
        or proof.get("evaluator_profile_schema_hash") != PROFILE_HASH
        or any(type(proof.get(k)) is not int or proof[k] != 0 for k in FORBIDDEN)
    ):
        raise ValueError("pure runtime isolation proof failed")
    calls = proof.get("learned_eval_calls")
    if (
        type(calls) is not int
        or calls < 0
        or type(response.get("nodes")) is not int
        or response["nodes"] < 0
    ):
        raise ValueError("invalid learned-work counters")
    if nodes is not None and response["nodes"] > nodes:
        raise ValueError("diagnostic node ceiling exceeded")
    if response.get("compute_control") is not None:
        raise ValueError("controller must be absent in this evaluator comparison")
    outcome, termination = response.get("outcome"), response.get("termination")
    if termination not in {"Completed", "Stable", "NodeLimit", "TimeLimit", "Cancelled"}:
        raise ValueError("unsupported or failed search termination")
    if outcome == "evaluated":
        if calls == 0 or response.get("legal") is not True or termination == "EvaluationError":
            raise ValueError("regular result lacks positive learned evaluation and legal move")
        if (
            type(response.get("score")) is not int
            or abs(response["score"]) > 30_000
            or any(
                type(response.get(k)) is not int or response[k] < 0 for k in ("depth", "seldepth")
            )
            or not isinstance(response.get("best_move"), str)
        ):
            raise ValueError("invalid evaluated score, depth or move")
        return
    if calls != 0 or any(response.get(k) != 0 for k in ("nodes", "depth", "seldepth")):
        raise ValueError("terminal/pre-evaluation result contains unexplained work")
    if any(proof.get(k) != 0 for k in ("accumulator_updates", "accumulator_refreshes")):
        raise ValueError("zero-evaluation outcome contains accumulator work")
    terminal = {"checkmate", "no_legal_moves", "repetition", "perpetual_check"}
    if outcome in terminal:
        if (
            response.get("best_move") is not None
            or termination != "Completed"
            or response.get("game_end") == "None"
        ):
            raise ValueError("invalid rule-terminal response")
        expected_score = 0 if outcome == "repetition" else -30_000
        if response.get("score") != expected_score and not (
            outcome == "perpetual_check" and response.get("score") == 30_000
        ):
            raise ValueError("invalid terminal rule score")
        return
    permitted = {
        "cancelled_before_evaluation": "Cancelled",
        "time_limit_before_evaluation": "TimeLimit",
        "node_limit_before_evaluation": "NodeLimit",
    }
    if (
        permitted.get(outcome) != termination
        or response.get("legal") is not True
        or response.get("score") != 0
        or not isinstance(response.get("best_move"), str)
    ):
        raise ValueError("invalid pre-evaluation interruption")


def _clock_limits(sfen: str, remaining_ns: int) -> tuple[int, int]:
    """Mirror the pinned common allocator only to enforce its independent host ceiling.

    Each timed response must echo the same Rust hard limit, making drift an error.
    The engine's 100ms reserve covers transport/validation/native move application.
    """
    remaining_ms = max(0, remaining_ns // NS_PER_MS)
    moves = (int(sfen.split()[3]) - 1) // 2
    expected = max(24, min(100, 100 - moves))
    allocated = min(max(1, remaining_ms // expected) * 3, remaining_ms)
    hard = allocated - min(100, max(0, allocated - 1))
    return allocated, hard


def _seal(path: Path, result: dict) -> dict:
    result = {**result, "receipt_sha256": hashlib.sha256(encoded(result)).hexdigest()}
    atomic(path, encoded(result))
    return result


def _read_receipt(root: Path, path: Path, plan_sha: str) -> dict:
    path = _inside(root, path)
    result = json.loads(path.read_text())
    signature = result.pop("receipt_sha256")
    if (
        result["plan_sha256"] != plan_sha
        or hashlib.sha256(encoded(result)).hexdigest() != signature
    ):
        raise ValueError("arena receipt identity changed")
    result["receipt_sha256"] = signature
    for ref in [result["trace"], *result.get("stderr_artifacts", [])]:
        if ref is not None:
            _reference(root, ref["path"], ref["sha256"])
    return result


def _stopping(output: Path) -> bool:
    return (output / "STOP").exists() or (output.parent / "STOP").exists()


def _play_game(
    root: Path,
    output: Path,
    spec: dict,
    plan: dict,
    plan_sha: str,
    folder: Path,
    helper: Any,
    resumed: dict | None,
    campaign_deadline: float,
) -> dict:
    players, events, processes = {}, 0, []
    initial_ns = spec["clock_ms"] * NS_PER_MS
    state = {
        "moves": [],
        "remaining_ns": {"black": initial_ns, "white": initial_ns},
        "expected_sfen": spec["initial_sfen"],
    }
    if resumed is not None:
        state = {key: resumed[key] for key in state}
        state["moves"] = state["moves"].copy()
        state["remaining_ns"] = state["remaining_ns"].copy()
    result = {
        **spec,
        **state,
        "plan_sha256": plan_sha,
        "status": "running",
        "score_candidate": None,
        "absolute_deadline_violations": 0,
        "resumed_from": resumed.get("receipt_sha256") if resumed else None,
    }
    journal = helper.TraceJournal(folder / "events.jsonl.gz")
    began, cpu_before = time.monotonic(), helper.cpu_usage()
    active = "black"
    pending_turn_started = None

    def record(event: dict, kind: str, **fields: Any) -> None:
        nonlocal events
        events += 1
        journal.write(
            helper.canonical(
                {"game_id": spec["id"], "event": events, "kind": kind, **fields, **event}
            )
        )
        journal.flush()

    def ask(player: Any, request: dict, deadline_ns: int, kind: str) -> dict:
        try:
            event = player.exchange(request, max(0, deadline_ns - time.monotonic_ns()))
        except helper.ProbeError as error:
            record(error.event, kind, failure=error.kind, side=active)
            raise
        record(event, kind, side=active)
        return event["response"]

    try:
        for role in ("baseline", "candidate"):
            asset = plan["artifacts"][role]
            argv = [
                str(root / plan["artifacts"]["probe"]["path"]),
                str(root / asset["path"]),
                asset["sha256"],
                "probe",
            ]
            players[role] = helper.Probe(argv, role, folder)
            response = ask(
                players[role],
                {
                    "sfen": spec["initial_sfen"],
                    "moves": state["moves"],
                    "depth": 1,
                    "nodes": 0,
                    "control": False,
                },
                time.monotonic_ns() + 30_000_000_000,
                "preparation",
            )
            _validate_response(response, asset["sha256"], nodes=0)
            if (
                response["sfen"] != state["expected_sfen"]
                or helper.adjudicated(response) is not None
            ):
                raise ValueError("prepared player restored a wrong or terminal root")
        result["preparation_seconds"] = time.monotonic() - began
        turn_started = time.monotonic_ns()
        while len(state["moves"]) < plan["config"]["max_plies"]:
            if _stopping(output) or time.monotonic() >= campaign_deadline:
                result.update(
                    status="stopped",
                    reason="requested_stop" if _stopping(output) else "campaign_wall_limit",
                )
                break
            active = "white" if state["expected_sfen"].split()[1] == "w" else "black"
            role = "candidate" if active == spec["candidate_side"] else "baseline"
            pending_turn_started = turn_started
            before = state["remaining_ns"].copy()
            now = time.monotonic_ns()
            effective = before[active] - (now - turn_started)
            if effective <= 0:
                result.update(
                    status="completed", reason="time_forfeit", winner=helper.opposite(active)
                )
                state["remaining_ns"][active] = 0
                break
            allocated_ms, hard_ms = _clock_limits(state["expected_sfen"], effective)
            deadline = now + allocated_ms * NS_PER_MS
            clocks = before.copy()
            clocks[active] = effective
            request = {
                "sfen": spec["initial_sfen"],
                "moves": state["moves"].copy(),
                "depth": 64,
                "black_time_ms": clocks["black"] // NS_PER_MS,
                "white_time_ms": clocks["white"] // NS_PER_MS,
                "control": False,
            }
            response = ask(players[role], request, deadline, "search")
            _validate_response(response, plan["artifacts"][role]["sha256"])
            if (
                response["sfen"] != state["expected_sfen"]
                or response["perspective"].lower() != active
                or not isinstance(response.get("time_hard_limit_ms"), (int, float))
                or not math.isclose(
                    response["time_hard_limit_ms"], hard_ms, rel_tol=0, abs_tol=1e-6
                )
                or helper.adjudicated(response) is not None
            ):
                raise ValueError("searched root, perspective or common clock plan mismatch")
            proposed = [*state["moves"], response["best_move"]]
            applied = ask(
                players["baseline"],
                {
                    "sfen": spec["initial_sfen"],
                    "moves": proposed,
                    "depth": 1,
                    "nodes": 0,
                    "control": False,
                },
                deadline,
                "rules_application",
            )
            _validate_response(applied, plan["artifacts"]["baseline"]["sha256"], nodes=0)
            terminal = helper.adjudicated(applied)
            state["moves"], state["expected_sfen"] = proposed, applied["sfen"]
            applied_ns = time.monotonic_ns()
            charged_ns = math.ceil((applied_ns - turn_started) / NS_PER_MS) * NS_PER_MS
            state["remaining_ns"][active] -= charged_ns
            pending_turn_started = None
            record(
                {},
                "move_applied",
                side=active,
                charged_ns=charged_ns,
                clock_before_ns=before,
                clock_after_ns=state["remaining_ns"].copy(),
                absolute_deadline_ns=deadline,
                applied_ns=applied_ns,
                engine_target_ms=response.get("time_target_ms"),
                engine_hard_limit_ms=hard_ms,
            )
            if applied_ns > deadline:
                result["absolute_deadline_violations"] += 1
                result.update(status="invalid", reason="absolute_deadline_violation", winner=None)
                break
            if state["remaining_ns"][active] <= 0:
                result.update(
                    status="completed", reason="time_forfeit", winner=helper.opposite(active)
                )
                break
            if terminal is not None:
                result.update(status="completed", **terminal)
                break
            atomic(folder / "confirmed-state.json", encoded({"plan_sha256": plan_sha, **state}))
            atomic(
                output / "progress.json",
                encoded(
                    {
                        "status": "running",
                        "game": spec["id"],
                        "plies": len(state["moves"]),
                        "remaining_ns": state["remaining_ns"],
                        "updated_unix_s": time.time(),
                    }
                ),
            )
            # Accounting for this journal/state publication begins the opponent's turn.
            turn_started = applied_ns
        else:
            result.update(status="incomplete", reason="max_plies_unscored", winner=None)
    except helper.ProbeError as error:
        if error.kind == "deadline":
            result["absolute_deadline_violations"] += 1
            result.update(
                status="invalid",
                reason="absolute_deadline_violation",
                error=str(error),
                winner=None,
            )
        elif (
            isinstance(error.__cause__, (EOFError, OSError))
            or error.event.get("process_returncode") is not None
        ):
            result.update(
                status="process_failure",
                reason="player_process_failure",
                error=str(error),
                winner=None,
            )
        else:
            result.update(
                status="invalid", reason="invalid_probe_protocol", error=str(error), winner=None
            )
    except (OSError, ValueError, KeyError, TypeError) as error:
        result.update(
            status="invalid",
            reason="contract_failure",
            error=f"{type(error).__name__}: {error}",
            winner=None,
        )
    finally:
        if pending_turn_started is not None and result["status"] not in {"completed", "stopped"}:
            elapsed = time.monotonic_ns() - pending_turn_started
            state["remaining_ns"][active] -= math.ceil(elapsed / NS_PER_MS) * NS_PER_MS
        for player in players.values():
            processes.append(player.close())
        result.update(state)
        result["processes"] = processes
        result["stderr_artifacts"] = [
            _reference(root, p["stderr_path"], p["stderr_sha256"]) for p in processes
        ]
        result["trace"] = journal.close()
        result["event_count"] = events
        result["wall_seconds"] = time.monotonic() - began
        result["child_cpu"] = helper.cpu_delta(cpu_before, helper.cpu_usage())
    if result["status"] != "invalid" and any(
        p["returncode"] != 0 and p["shutdown"] == "stdin_eof" for p in processes
    ):
        result.update(status="process_failure", reason="player_process_failure", winner=None)
    if result["status"] == "completed":
        result["score_candidate"] = (
            0.5 if result["winner"] is None else float(result["winner"] == spec["candidate_side"])
        )
    return _seal(folder / "receipt.json", result)


def _summary(plan: dict, games: list[dict], attempts: list[dict]) -> dict:
    scored = [g for g in games if g["group"] == "evaluation"]
    complete = len(scored) == 32 and all(g["status"] == "completed" for g in scored)
    planned = 32 + plan["criteria"]["demonstration_games"]
    all_complete = len(games) == planned and all(g["status"] == "completed" for g in games)
    adverse = [
        a
        for a in attempts
        if a["status"] in {"invalid", "process_failure", "incomplete"}
        or a.get("absolute_deadline_violations", 0)
        or a.get("reason") == "time_forfeit"
    ]
    scores, lower = {}, None
    if complete:
        for clock_ms in (180_000, 600_000):
            group = [g["score_candidate"] for g in scored if g["clock_ms"] == clock_ms]
            scores[str(clock_ms)] = sum(group) / len(group)
        clusters = sorted({str(g.get("family", g["pair"])) for g in scored})
        pair_scores = np.array(
            [
                np.mean(
                    [
                        g["score_candidate"]
                        for g in scored
                        if str(g.get("family", g["pair"])) == cluster
                    ]
                )
                for cluster in clusters
            ]
        )
        rng = np.random.default_rng(plan["criteria"]["bootstrap_seed"])
        draws = rng.integers(0, len(clusters), size=(10_000, len(clusters)))
        lower = float(np.percentile(pair_scores[draws].mean(axis=1), 5))
    strata = (
        {
            s: float(np.mean([g["score_candidate"] for g in scored if g.get("stratum") == s]))
            for s in sorted({g["stratum"] for g in scored if "stratum" in g})
        }
        if complete
        else {}
    )
    adopted = (
        all_complete
        and not adverse
        and complete
        and all(s > 0.5 for s in scores.values())
        and lower > 0.5
        and all(s >= plan["criteria"].get("minimum_stratum_score", 0) for s in strata.values())
    )
    return {
        "evaluation_complete": complete,
        "all_planned_complete": all_complete,
        "completed_games": sum(g["status"] == "completed" for g in games),
        "unscored_games": sum(g["status"] != "completed" for g in games),
        "unplayed_games": planned - len(games),
        "adverse_attempts": len(adverse),
        "candidate_score_by_clock": scores,
        "paired_bootstrap_one_sided_95_lower": lower,
        "source_family_count": len({g.get("family", g["pair"]) for g in scored}),
        "stratum_scores": strata,
        "demonstration_scores": [
            g.get("score_candidate") for g in games if g["group"] == "demonstration"
        ],
        "demonstrations_are_independent_strength_evidence": False,
        "adoption_criteria_met": bool(adopted),
        "promotion_performed": False,
        "human_shodan_validation": False,
    }


def run_arena(
    root: Path, output: Path, config: dict, *, pause_after_games: int | None = None
) -> dict:
    """Execute/resume 40 scheduled games, retaining completed games and all failed attempts."""
    root = root.resolve(strict=True)
    output = _inside(root, output, file=False)
    if not output.is_relative_to(root / "local"):
        raise ValueError("arena output must remain under repository local/")
    plan = _plan(root, config)
    if pause_after_games is not None and (
        type(pause_after_games) is not int
        or not 2 <= pause_after_games <= len(plan["games"])
        or pause_after_games % 2
    ):
        raise ValueError("pause bound must preserve color pairs within the fixed schedule")
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "plan.json"
    if plan_path.exists():
        _inside(root, plan_path)
        if plan_path.read_bytes() != encoded(plan):
            raise ValueError("arena identity or acceptance plan changed; new run required")
    elif any(output.iterdir()):
        raise ValueError("refuse an unowned nonempty arena directory")
    else:
        atomic(plan_path, encoded(plan))
    plan_sha = digest(plan_path)
    spec = importlib.util.spec_from_file_location(
        "evaluator_arena_probe", root / "scripts/compare_core_prototype.py"
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    attempts, games, status = [], [], "complete"
    campaign_deadline = time.monotonic() + plan["max_wall_seconds"]
    for game in plan["games"]:
        if pause_after_games is not None and len(games) >= pause_after_games:
            status = "paused"
            break
        game_folder = output / "games" / game["id"]
        _inside(root, game_folder, file=False)
        game_folder.mkdir(parents=True, exist_ok=True)
        previous = []
        for folder in sorted(game_folder.glob("attempt-*")):
            _inside(root, folder, file=False)
            receipt = folder / "receipt.json"
            if not receipt.exists():
                trace = folder / "events.jsonl.gz"
                orphan = {
                    **game,
                    "plan_sha256": plan_sha,
                    "status": "process_failure",
                    "reason": "unclosed_process_attempt",
                    "score_candidate": None,
                    "trace": _reference(root, trace) if trace.exists() else None,
                    "stderr_artifacts": [],
                    "absolute_deadline_violations": 0,
                    "decompression_verified": False,
                    "budget_charge_seconds": game["clock_ms"] * 2 / 1000 + 60,
                    "budget_charge_basis": "conservative upper bound for unclosed attempt",
                }
                _seal(receipt, orphan)
            saved = _read_receipt(root, receipt, plan_sha)
            if any(saved.get(key) != value for key, value in game.items()):
                raise ValueError("arena attempt belongs to another scheduled game")
            previous.append(saved)
        attempts.extend(previous)
        campaign_deadline -= sum(
            a.get("budget_charge_seconds", a.get("wall_seconds", 0.0)) for a in previous
        )
        if previous and previous[-1]["status"] in {"completed", "incomplete", "invalid"}:
            games.append(previous[-1])
            if previous[-1]["status"] == "invalid":
                status = "failed"
                break
            continue
        if sum(a["status"] == "process_failure" for a in previous) > 1:
            games.append(previous[-1])
            status = "failed"
            break
        while True:
            if _stopping(output) or time.monotonic() >= campaign_deadline:
                status = "stopped"
                break
            folder = game_folder / f"attempt-{len(previous):03d}"
            folder.mkdir(exist_ok=False)
            resumed = previous[-1] if previous and previous[-1]["status"] == "stopped" else None
            result = _play_game(
                root, output, game, plan, plan_sha, folder, helper, resumed, campaign_deadline
            )
            previous.append(result)
            attempts.append(result)
            if (
                result["status"] == "process_failure"
                and sum(a["status"] == "process_failure" for a in previous) == 1
            ):
                continue
            games.append(result)
            status = (
                "stopped"
                if result["status"] == "stopped"
                else "failed"
                if result["status"] in {"invalid", "process_failure"}
                else "complete"
            )
            break
        if status != "complete":
            break
    for ref in plan["artifacts"].values():
        _reference(root, ref["path"], ref["sha256"])
    summary = _summary(plan, games, attempts)
    result = {
        "schema": SCHEMA,
        "status": status,
        "plan_sha256": plan_sha,
        "planned_games": len(plan["games"]),
        "evaluation_games": 32,
        "demonstration_games": plan["criteria"]["demonstration_games"],
        "games": games,
        "attempts": attempts,
        "summary": summary,
        "adoption_criteria_met": summary["adoption_criteria_met"],
    }
    atomic(output / "arena.json", encoded(result))
    return result
