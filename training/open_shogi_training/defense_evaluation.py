"""Held-out move-quality screen; the offline teacher never participates in a game."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path

from .defense_scenarios import GROUPS, R3Probe
from .evaluator_data import Replay, _teacher_observation, atomic, digest, encoded, teacher_config
from .labeling.usi import USIEngine


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
        rows = [r for r in cases if r["group"] == group and r["eligible"]]
        metrics[group] = {
            "eligible": len(rows),
            "excluded_lost_or_mate": sum(r["group"] == group for r in cases) - len(rows),
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
        "screen_pass": bool(improved and preserved),
    }


def screen(root: Path, run: Path, config: dict) -> dict:
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
                    cases.append(case)
                    continue
                state = replay.ask(reset=row["sfen"], successors=True)
                searches = {
                    name: probe.search(row["sfen"], [], row["sfen"])
                    for name, probe in probes.items()
                }
                reference, terminal = _teacher_observation(
                    teacher,
                    state,
                    root=root,
                    output=output,
                    config=generation,
                    game=row["game"],
                    ply=row["ply"],
                    branch="screen-root",
                    moves=[],
                    records=[],
                )
                values, observations = {}, {}
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
                                    30000 if f"winner: {winner}" in child["terminal"] else -30000
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
                        result, ending = _teacher_observation(
                            teacher,
                            child_state,
                            root=root,
                            output=output,
                            config=generation,
                            game=row["game"],
                            ply=row["ply"] + 1,
                            branch="screen-child",
                            moves=[move],
                            records=[],
                        )
                        if ending is not None:
                            values[move] = -30000 if ending["outcome"] == "win" else 30000
                            observations[move] = {"validated_terminal": ending}
                        else:
                            observations[move] = [c.as_dict() for c in result.candidates]
                            if result.primary.score.kind == "cp":
                                values[move] = -result.primary.score.value
                            elif result.primary.score.kind == "mate":
                                values[move] = -30000 if result.primary.score.value > 0 else 30000
                eligible = (
                    bool(values)
                    and all(s["best_move"] in values for s in searches.values())
                    and max(values.values()) > -1500
                )
                case = {
                    "plan_sha256": digest(plan_path),
                    "group": row["group"],
                    "family": row["family"],
                    "game": row["game"],
                    "sfen": row["sfen"],
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
    report = {
        "schema": "open_shogiai_defense_screen/v1",
        "plan_sha256": digest(plan_path),
        "cases": len(cases),
        "unknown_cases": sum(c["group"] != "known_regression" for c in cases),
        "known_regressions": [
            {"sfen": c["sfen"], "eligible": c["eligible"], "regret_cp": c["regret_cp"]}
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
