"""Held-out move-quality screen; the offline teacher never participates in a game."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import sqlite3
from pathlib import Path

from .defense_scenarios import GROUPS, R3Probe
from .evaluator_data import Replay, _teacher_observation, atomic, digest, encoded, teacher_config
from .evaluator_ledger import DeferredTaskError, task_identity
from .labeling.usi import USIEngine, USIIncompleteDepthError, USITimeoutError


def select_roots(dataset: Path, seed: int, per_group: int = 16) -> list[dict]:
    manifest = json.loads((dataset / "manifest.json").read_text())
    ref = next(r for r in manifest["artifacts"] if r["path"] == "development_test-rows.jsonl.gz")
    path = dataset / ref["path"]
    if path.is_symlink() or digest(path) != ref["sha256"]:
        raise ValueError("held-out rows changed")
    games = {g["game"] for g in manifest["source_games"] if g["split"] == "development_test"}
    pools = {group: [] for group in GROUPS}
    with gzip.open(path, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if row["game"] not in games:
                raise ValueError("held-out lineage mismatch")
            lo, hi = {
                "general": (32, 128),
                "opening": (12, 40),
                "defense": (16, 96),
                "attack_end": (80, 191),
            }[row["group"]]
            if row["kind"] == "root" and lo <= row["ply"] <= hi:
                pools[row["group"]].append(row)
    selected = []
    for group, rows in pools.items():
        used = set()
        for row in sorted(rows, key=lambda r: hashlib.sha256(encoded([seed, r["sfen"]])).digest()):
            if row["game"] in used:
                continue
            selected.append(row)
            used.add(row["game"])
            if len(used) == per_group:
                break
        if len(used) != per_group:
            raise ValueError(f"need {per_group} held-out trajectories in {group}")
    return selected


def summarize(cases: list[dict]) -> dict:
    metrics = {}
    for group in GROUPS:
        selected = [r for r in cases if r["group"] == group]
        completed = [r for r in selected if r.get("status", "completed") == "completed"]
        rows = [r for r in completed if r["eligible"]]
        metrics[group] = {
            "planned": len(selected),
            "completed": len(completed),
            "missing": sum(r.get("status") == "missing" for r in selected),
            "not_run": sum(r.get("status") == "not_run" for r in selected),
            "eligible": len(rows),
            "excluded_lost_or_mate": len(completed) - len(rows),
        }
        for model in ("r3", "candidate"):
            regrets = [r["regret_cp"][model] for r in rows]
            metrics[group][model] = {
                "mean_regret_cp": sum(regrets) / len(regrets) if regrets else None,
                "major_errors_400cp": sum(v >= 400 for v in regrets),
                "reasonable_within_100cp": sum(v <= 100 for v in regrets),
            }
    enough = all(m["eligible"] >= 8 for m in metrics.values())
    improved = enough and all(
        metrics[g]["candidate"]["major_errors_400cp"] < metrics[g]["r3"]["major_errors_400cp"]
        and metrics[g]["candidate"]["mean_regret_cp"] <= 0.9 * metrics[g]["r3"]["mean_regret_cp"]
        for g in ("opening", "defense")
    )
    preserved = enough and all(
        metrics[g]["candidate"]["major_errors_400cp"] <= metrics[g]["r3"]["major_errors_400cp"]
        and metrics[g]["candidate"]["mean_regret_cp"] <= metrics[g]["r3"]["mean_regret_cp"] + 50
        for g in ("general", "attack_end")
    )
    return {
        "groups": metrics,
        "sufficient_cases": enough,
        "opening_defense_improved": bool(improved),
        "general_attack_preserved": bool(preserved),
        "screen_pass": bool(improved and preserved)
        if all(c.get("status", "completed") == "completed" for c in cases)
        else None,
    }


def _missing_teacher_evidence(error, output: Path, config: dict, request: dict) -> dict:
    """A deferred task is not itself proof of an optional depth/time-budget miss."""
    evidence = {"failure_kind": type(error).__name__, "requested_depth": config["teacher_depth"]}
    if isinstance(error, DeferredTaskError):
        key = str(error)
        # Read the existing evidence without creating or modifying a task ledger.
        with sqlite3.connect(f"file:{output / 'tasks.sqlite3'}?mode=ro", uri=True) as db:
            row = db.execute(
                "SELECT identity,status,attempts FROM tasks WHERE id=?", (key,)
            ).fetchone()
        identity = task_identity(
            config,
            request["game"],
            request["ply"],
            request["branch"],
            request["state"],
            request["moves"],
        )
        if row is None or json.loads(row[0]) != identity or row[1] not in {"hard", "deferred"}:
            raise ValueError("optional screen deferred evidence mismatch") from error
        attempts = json.loads(row[2])
        allowed = {"USIIncompleteDepthError", "USITimeoutError"}
        budget = isinstance(error.__cause__, DeferredTaskError) and str(error.__cause__) in {
            "cumulative teacher attempt budget exhausted",
            "task wall budget exhausted",
            "hard queue budget exhausted",
        }
        if (
            error.__cause__ is not None
            and not budget
            and not isinstance(error.__cause__, (USIIncompleteDepthError, USITimeoutError))
        ):
            raise error
        if any(a["outcome"] not in allowed for a in attempts) or not (attempts or budget):
            raise error
        evidence.update(
            ledger_task=key,
            task_identity=identity,
            attempts=attempts,
            failure_kind=attempts[-1]["outcome"] if attempts else "finite_budget_exhausted",
            finite_budget_exhausted=budget,
        )
    else:
        evidence.update(
            node_ceiling=getattr(error, "node_ceiling", config["teacher_nodes"]),
            error=str(error),
        )
        receipt = getattr(error, "failure_receipt", None)
        if receipt is not None:
            evidence.update(failure_receipt=receipt, failure_sha256=digest(output / receipt))
    return {
        **evidence,
        "ply": request["ply"],
        "sfen": request["state"]["sfen"],
        "branch": request["branch"],
        "moves": request["moves"],
    }


def _case_identity(row: dict, plan_sha: str) -> dict:
    return {
        "plan_sha256": plan_sha,
        **{key: row[key] for key in ("group", "family", "game", "sfen", "ply")},
        "split": "development_test" if row["group"] != "known_regression" else "known_regression",
    }


def _report(output: Path, plan_sha: str, cases: list[dict], mode: str) -> dict:
    counts = {
        s: sum(c.get("status", "completed") == s for c in cases)
        for s in ("completed", "missing", "not_run")
    }
    report = {
        "schema": "open_shogiai_defense_screen/v1",
        "plan_sha256": plan_sha,
        "execution_mode": mode,
        "status": "completed"
        if counts["completed"] == len(cases)
        else "partial"
        if counts["completed"]
        else "missing"
        if counts["missing"]
        else "not_run",
        "cases": len(cases),
        **counts,
        "unknown_cases": sum(c["group"] != "known_regression" for c in cases),
        "unavailable_cases": [
            {key: c[key] for key in ("group", "family", "game", "ply", "split", "status")}
            for c in cases
            if c.get("status") in {"missing", "not_run"}
        ],
        "known_regressions": [
            {
                "sfen": c["sfen"],
                "eligible": c["eligible"],
                "regret_cp": c["regret_cp"],
                "status": c.get("status", "completed"),
            }
            for c in cases
            if c["group"] == "known_regression"
        ],
        **summarize(cases),
        "teacher_is_reference_not_ground_truth": True,
        "known_regressions_are_separate": True,
        "comparison": "100k-node offline screen; real 3/10-minute games are separate",
        "human_shodan_validated": False,
    }
    atomic(output / "result.json", encoded(report))
    return report


def screen(root: Path, run: Path, config: dict) -> dict:
    mode = config.get("_optional_screen", "execute")
    if mode not in {"execute", "retained_only"}:
        raise ValueError("unknown optional screen mode")
    output = run / "move-screen"
    output.mkdir(exist_ok=True)
    generation = copy.deepcopy(config["generation"])
    generation["teacher_depth"] = config["evaluation"]["screen_teacher_depth"]
    generation["teacher_nodes"] = 2_000_000
    generation["defense_campaign"]["probe_nodes"] = config["evaluation"]["screen_probe_nodes"]
    roots = select_roots(run / "data/dataset", config["seed"])
    known_path = root / config["evaluation"]["regression_positions_path"]
    if (
        known_path.is_symlink()
        or digest(known_path) != config["evaluation"]["regression_positions_sha256"]
    ):
        raise ValueError("known regression inputs changed")
    roots.extend(json.loads(known_path.read_text())["roots"])
    plan = {
        "run_sha256": digest(run / "run.json"),
        "candidate_sha256": digest(run / "fit/best.osaval03"),
        "roots": roots,
        "config": config["evaluation"],
    }
    plan_path = output / "plan.json"
    if plan_path.exists() and plan_path.read_bytes() != encoded(plan):
        raise ValueError("move screen plan changed")
    atomic(plan_path, encoded(plan))
    plan_sha = digest(plan_path)
    if mode == "retained_only":
        cases = []
        for i, row in enumerate(roots):
            case = _case_identity(row, plan_sha)
            path = output / f"case-{i:03d}.json"
            if path.exists():
                saved = json.loads(path.read_text())
                checksum = saved.pop("sha256")
                if (
                    hashlib.sha256(encoded(saved)).hexdigest() != checksum
                    or saved["plan_sha256"] != plan_sha
                ):
                    raise ValueError("move screen receipt changed")
                case.update({"status": "completed", **saved})
            else:
                case.update(status="not_run", eligible=False, regret_cp=None)
            cases.append(case)
        return _report(output, plan_sha, cases, mode)
    candidate_config = copy.deepcopy(generation)
    candidate_config.update(
        leaf_path=str((run / "fit/best.osaval03").relative_to(root)),
        leaf_sha256=plan["candidate_sha256"],
    )
    # Searches run sequentially; no training or teacher work overlaps a measured probe.
    probes = {"r3": R3Probe(root, generation), "candidate": R3Probe(root, candidate_config)}
    replay = Replay(root, generation)
    cases = []
    try:
        with USIEngine(
            teacher_config(root, generation),
            root,
            isolate_process_group=False,
            allow_terminal_outcomes=True,
        ) as teacher:
            for i, row in enumerate(roots):
                if (run / "STOP").exists():
                    raise InterruptedError("move screen stopped")
                path = output / f"case-{i:03d}.json"
                if path.exists():
                    case = json.loads(path.read_text())
                    checksum = case.pop("sha256")
                    if hashlib.sha256(encoded(case)).hexdigest() != checksum or case[
                        "plan_sha256"
                    ] != digest(plan_path):
                        raise ValueError("move screen receipt changed")
                    cases.append({**_case_identity(row, plan_sha), "status": "completed", **case})
                    continue
                state = replay.ask(reset=row["sfen"], successors=True)
                searches = {
                    name: probe.search(row["sfen"], [], row["sfen"])
                    for name, probe in probes.items()
                }
                legal = {c["move"]: c for c in state["successors"]}
                if any(s["best_move"] not in legal for s in searches.values()):
                    raise ValueError("screen proposed illegal move")
                values, observations, missing = {}, {}, []
                request = dict(
                    state=state, game=row["game"], ply=row["ply"], branch="screen-root", moves=[]
                )
                try:
                    reference, terminal = _teacher_observation(
                        teacher, root=root, output=output, config=generation, records=[], **request
                    )
                    if terminal is None:
                        choices = {c.pv[0] for c in reference.candidates} | {
                            s["best_move"] for s in searches.values()
                        }
                        legal = {c["move"]: c for c in state["successors"]}
                        for move in sorted(choices):
                            if move not in legal:
                                raise ValueError("screen proposed illegal move")
                            child = legal[move]
                            if child["terminal"] != "None":
                                # This is an actual game ending, never a handcrafted position value.
                                winner = "Black" if row["sfen"].split()[1] == "b" else "White"
                                if "winner:" in child["terminal"]:
                                    values[move] = (
                                        30000
                                        if f"winner: {winner}" in child["terminal"]
                                        else -30000
                                    )
                                elif "loser:" in child["terminal"]:
                                    values[move] = (
                                        -30000 if f"loser: {winner}" in child["terminal"] else 30000
                                    )
                                else:
                                    raise ValueError(
                                        "unsupported terminal screen outcome; retain evidence"
                                    )
                                observations[move] = {"native_terminal": child["terminal"]}
                                continue
                            child_state = replay.ask(reset=child["sfen"], successors=True)
                            request = dict(
                                state=child_state,
                                game=row["game"],
                                ply=row["ply"] + 1,
                                branch="screen-child",
                                moves=[move],
                            )
                            result, ending = _teacher_observation(
                                teacher,
                                root=root,
                                output=output,
                                config=generation,
                                records=[],
                                **request,
                            )
                            if ending is not None:
                                values[move] = -30000 if ending["outcome"] == "win" else 30000
                                observations[move] = {"validated_terminal": ending}
                            else:
                                observations[move] = [c.as_dict() for c in result.candidates]
                                if result.primary.score.kind == "cp":
                                    values[move] = -result.primary.score.value
                                elif result.primary.score.kind == "mate":
                                    values[move] = (
                                        -30000 if result.primary.score.value > 0 else 30000
                                    )
                except (DeferredTaskError, USIIncompleteDepthError, USITimeoutError) as error:
                    missing.append(_missing_teacher_evidence(error, output, generation, request))
                eligible = (
                    not missing
                    and bool(values)
                    and all(s["best_move"] in values for s in searches.values())
                    and max(values.values()) > -1500
                )
                case = {
                    **_case_identity(row, plan_sha),
                    "status": "missing" if missing else "completed",
                    "missing_observations": missing,
                    "eligible": eligible,
                    "searches": searches,
                    "teacher_children": observations,
                    "regret_cp": {
                        name: max(values.values()) - values[s["best_move"]]
                        for name, s in searches.items()
                    }
                    if eligible
                    else None,
                }
                atomic(path, encoded({**case, "sha256": hashlib.sha256(encoded(case)).hexdigest()}))
                cases.append(case)
    finally:
        replay.close()
        for probe in probes.values():
            probe.close()
    return _report(output, plan_sha, cases, mode)
