"""C4 finite generations inside the existing leased evaluator runner."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path

import torch

from .evaluator_data import atomic, digest, encoded, symmetry_keys
from .evaluator_training import evaluate_groups, grouped_arrays, train
from .phase10v_model import Phase10VModel, torch_parameters
from .r4_c4_data import bind, build, generate, read, stopped
from .r4_data import reference


def validate(config):
    p = config["iteration"]
    if (
        p["schema"] != "open_shogiai_c4_iteration/v1"
        or not 3 <= p["minimum_generations"] <= p["maximum_generations"] <= 5
    ):
        raise ValueError("invalid finite C4 generation count")
    if not 1 <= p["games_per_generation"] <= 16384 or p["bootstrap"]["games"] > 32:
        raise ValueError("invalid C4 trajectory bound")
    if p["generation"]["teacher"]["strong_depth"] <= p["generation"]["teacher"]["broad_depth"]:
        raise ValueError("C4 strong reanalysis must be distinct")
    if any(
        not 0 < v <= 1 for v in config["training"]["coverage_sampler"]["origin_fractions"].values()
    ):
        raise ValueError("invalid origin sampling quota")
    if abs(sum(config["training"]["coverage_sampler"]["origin_fractions"].values()) - 1) > 1e-9:
        raise ValueError("origin quotas must sum to one")
    if {s["split"] for s in p["generation"]["starts"]} != {
        "train",
        "validation",
        "development_test",
    }:
        raise ValueError("missing fixed generation partitions")
    for name, lower, upper in (
        ("nodes", 128, 65536),
        ("max_plies", 32, 512),
        ("sample_stride", 1, 16),
        ("explore_stride", 2, 32),
        ("explore_nodes", 128, 65536),
        ("strong_roots_per_game", 1, 32),
        ("reply_plies", 0, 4),
        ("explore_tolerance_cp", 0, 150),
    ):
        if not lower <= p["generation"][name] <= upper:
            raise ValueError(f"unbounded C4 {name}")
    expected = {
        "actor_minimum_mean_clock_score": 0.55,
        "actor_minimum_each_clock_score": 0.5,
        "actor_minimum_cluster_lower": 0.45,
        "scalar_regression_ratio": 1.03,
    }
    if any(p["selection"][k] != v for k, v in expected.items()):
        raise ValueError("C4 actor policy differs from reviewed transition")
    teacher = p["generation"]["teacher"]
    if not (
        1 <= teacher["broad_depth"] < teacher["strong_depth"] <= 16
        and 1 <= teacher["broad_nodes"] <= teacher["strong_nodes"] <= 4_000_000
        and 1000 <= teacher["timeout_ms"] <= 60000
        and 0 <= teacher["maximum_worker_exits"] <= 3
    ):
        raise ValueError("unbounded C4 teacher work")
    if not 1 <= config["training"]["max_steps"] <= 24576:
        raise ValueError("unbounded C4 optimizer updates")


def training_config(root, config, initial, spec):
    result = {**copy.deepcopy(config["training"]), **spec.get("training_overrides", {})}
    result.update(
        seed=config["seed"] + spec["index"],
        initial_model=str(root / initial["path"]),
        initial_sha256=initial["sha256"],
    )
    result["stop_path"] = str(root / config["output"] / "STOP")
    if spec.get("stagnation", 0) >= 2:
        result["learning_rate"] *= 0.5
        result["minimum_learning_rate"] *= 0.5
        result["coverage_sampler"]["origin_fractions"] = {
            "0": 0.35,
            "4": 0.35,
            "5": 0.25,
            "6": 0.05,
        }
    return result


def identity(run, config, folder):
    return {
        "code_commit": config["code"]["commit"],
        "run_sha256": digest(run / "run.json"),
        "dataset_sha256": digest(folder / "data/dataset/manifest.json"),
    }


def changed(initial: Path, candidate: Path):
    before, after = Phase10VModel.read(initial), Phase10VModel.read(candidate)
    a, b = torch_parameters(before), torch_parameters(after)
    counts = [int(torch.count_nonzero(x.detach() != y.detach())) for x, y in zip(a, b, strict=True)]
    if not any(counts) or before.sha256 == after.sha256:
        raise ValueError("C4 candidate has no quantized tensor update")
    return counts


def audit(root, run, config, model, output):
    from .evaluator_run import _audit_report

    output.mkdir(parents=True, exist_ok=True)
    path = output / "model-audit.json"
    if not path.exists():
        subprocess.run(
            [
                "node",
                str(root / "scripts/check_evaluator_model.mjs"),
                str(root / config["runtime"]["module"]["path"]),
                str(model),
                digest(model),
                str(root / config["runtime"]["replay"]["path"]),
                str(path),
            ],
            cwd=root,
            check=True,
            timeout=180,
        )
    return _audit_report(run, config, report=path, model=model)


def compare(root, run, config, folder, candidate, baseline, dataset, confirmation=None):
    from .evaluator_arena import run_arena

    # All weights use the same captured runtime, clocks, start families and OFF ponder.
    plan = {
        **config["evaluation"],
        "seed": config["seed"],
        "probe_path": config["runtime"]["probe"]["path"],
        "probe_sha256": config["runtime"]["probe"]["sha256"],
        "baseline_path": baseline["path"],
        "baseline_sha256": baseline["sha256"],
        "candidate_path": candidate["path"],
        "candidate_sha256": candidate["sha256"],
        "dataset_path": str(dataset.relative_to(root)),
    }
    if confirmation:
        plan["c4_confirmation_starts"] = confirmation
    result = run_arena(root, folder, plan)
    if result["status"] == "stopped":
        raise InterruptedError("arena stopped with retained game receipts")
    if result["status"] not in {"complete", "failed"}:
        raise ValueError("unknown C4 arena state")
    return result


def confirmation_starts(root, run, config, initial):
    """New final series are created only after candidate selection is immutable."""
    output = run / "confirmation"
    path = output / "starts.json"
    if path.exists():
        return reference(root, path)
    if (output / "missing.json").exists():
        return None
    if not (run / "selection.json").exists():
        raise ValueError("freeze candidate selection before final confirmation")
    spec = {
        "id": "confirmation",
        "index": 100,
        "games": config["iteration"]["confirmation_games"],
        "generation_overrides": {"max_plies": 128, "strong_roots_per_game": 0, "reply_plies": 0},
    }
    confirmation_config = copy.deepcopy(config)
    confirmation_config["iteration"]["generation"]["starts"] = config["iteration"][
        "confirmation_starts"
    ]
    generated = generate(root, run, output / "trajectories", confirmation_config, initial, spec)
    db = sqlite3.connect(f"file:{run / 'position-index.sqlite3'}?mode=ro", uri=True)
    pools = {g: [] for g in ("general", "opening", "defense", "attack_end")}
    seen = set()
    try:
        for ref in generated["artifacts"]:
            raw = json.loads(gzip.decompress((root / ref["path"]).read_bytes()))
            options = sorted(
                raw["observations"][16:], key=lambda r: hashlib.sha256(r["sfen"].encode()).digest()
            )
            for r in options:
                key = min(symmetry_keys(r["sfen"]))
                if (
                    key not in seen
                    and not db.execute("SELECT 1 FROM owners WHERE key=?", (key,)).fetchone()
                ):
                    pools[raw["group"]].append(
                        {
                            "sfen": r["sfen"],
                            "game": raw["game"],
                            "family": raw["family"],
                            "group": raw["group"],
                            "ply": int(r["sfen"].split()[-1]) - 1,
                        }
                    )
                    seen.add(key)
                    break
        if any(len(v) < 8 for v in pools.values()):
            bind(
                output / "missing.json",
                {
                    "status": "missing",
                    "counts": {g: len(v) for g, v in pools.items()},
                    "fallback": "development comparison; novel confirmation unverified",
                    "selection_sha256": digest(run / "selection.json"),
                },
            )
            return None
        starts = [pools[g][i] for i in range(8) for g in pools]
        bind(
            path,
            {
                "starts": starts,
                "selection_sha256": digest(run / "selection.json"),
                "purpose": "final only; no training or candidate reselection",
            },
        )
        return reference(root, path)
    finally:
        db.close()


def transition(incumbent, actor, candidate, arena, groups, stagnation):
    """Internal actor admission only; this never changes the public default."""
    scores = arena["summary"]["candidate_score_by_clock"]
    preserved = all(
        groups["candidate"][g]["loss"] <= groups["baseline"][g]["loss"] * 1.03
        for g in groups["baseline"]
    )
    lower = arena["summary"]["paired_bootstrap_one_sided_95_lower"]
    admitted = (
        arena["status"] == "complete"
        and arena["summary"]["all_planned_complete"]
        and arena["summary"]["adverse_attempts"] == 0
        and scores
        and all(v >= 0.5 for v in scores.values())
        and sum(scores.values()) / len(scores) >= 0.55
        and lower is not None
        and lower >= 0.45
        and preserved
    )
    return {
        "incumbent": candidate if admitted else incumbent,
        "actor": candidate if admitted else actor,
        "trained_candidate": candidate,
        "actor_updated": bool(admitted),
        "stagnation": 0 if admitted else stagnation + 1,
        "scalar_preserved": preserved,
        "promotion_performed": False,
        "next_branch": "lower_lr_more_strong_and_replay"
        if not admitted and stagnation + 1 >= 2
        else "normal_new_experience",
    }


def snapshot(root, run):
    """Read checkpoint and immutable task receipts without replaying completed work."""
    result = {"games": 0, "rows": 0, "files": {}, "pending": {}, "tasks": {}, "counters": {}}
    for folder in [run, *sorted((run / "generations").glob("G*"))]:
        trajectories = folder / "data/trajectories"
        for receipt in sorted(trajectories.glob("*.receipt.json")):
            shard = receipt.with_name(receipt.name.replace(".receipt.json", ".gz"))
            saved = read(receipt)
            reference(root, shard, saved["sha256"])
            result["files"][str(receipt.relative_to(run))] = digest(receipt)
            result["files"][str(shard.relative_to(run))] = saved["sha256"]
            result["games"] += 1
            result["rows"] += saved["counts"]["label_rows"]
        ledger = trajectories / "tasks.sqlite3"
        if ledger.exists():
            db = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True)
            try:
                if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                    raise ValueError("C4 task ledger integrity failure")
                for key, payload, attempts in db.execute("SELECT id,identity,attempts FROM tasks"):
                    if hashlib.sha256(payload.encode()).hexdigest() != key:
                        raise ValueError("C4 task identity mismatch")
                    if len(json.loads(attempts)) > 2:
                        raise ValueError("C4 task retry budget exceeded")
                result["counters"][str(folder.relative_to(run))] = dict(
                    db.execute("SELECT name,value FROM counters")
                )
                result["tasks"][str(folder.relative_to(run))] = dict(
                    db.execute("SELECT status,count(*) FROM tasks GROUP BY status")
                )
            finally:
                db.close()
        for name in (
            "spec.json",
            "data/trajectories/result.json",
            "data/dataset/manifest.json",
            "decision.json",
        ):
            path = folder / name
            if path.exists():
                result["files"][str(path.relative_to(run))] = digest(path)
        progress = folder / "data/trajectories/progress.json"
        if progress.exists():
            values = read(progress)
            if len(list(trajectories.glob("*.receipt.json"))) < values.get("games", 0):
                raise ValueError("C4 committed games disappeared")
        path = folder / "fit/resume.json"
        if path.exists():
            ref = read(path)
            checkpoint = folder / "fit" / ref["path"]
            if checkpoint.parent != folder / "fit":
                raise ValueError("C4 checkpoint escapes generation")
            reference(root, checkpoint, ref["sha256"])
            result["checkpoint"] = {**ref, "generation": folder.name}
    state = read(run / "state.json") if (run / "state.json").exists() else {}
    if state.get("execution_attempt"):
        prefix = run / "attempts" / f"{state['execution_attempt']:06d}"
        previous = Path(f"{prefix}-pause.json")
        if not previous.exists():
            previous = prefix.with_suffix(".json")
        old = read(previous)["snapshot"]
        for name, sha in old["files"].items():
            reference(root, run / name, sha)
        for generation, counters in old["counters"].items():
            if any(
                result["counters"].get(generation, {}).get(k, -1) < v for k, v in counters.items()
            ):
                raise ValueError("C4 cumulative task budget decreased")
        for generation, tasks in old["tasks"].items():
            if result["tasks"].get(generation, {}).get("accepted", 0) < tasks.get("accepted", 0):
                raise ValueError("C4 accepted task cursor decreased")
    return result


def verify_results(root, run):
    report = read(run / "c4-result.json")
    reference(root, Path(report["candidate"]["path"]), report["candidate"]["sha256"])
    for path in sorted((run / "generations").glob("G*/decision.json")):
        decision = read(path)
        for ref in [decision["trained_candidate"], *decision["artifacts"]]:
            reference(root, Path(ref["path"]), ref["sha256"])
    if read(run / "selection.json")["candidate"] != report["candidate"]:
        raise ValueError("C4 selected export identity changed")
    for path in sorted((run / "final").glob("*/arena.json")):
        for attempt in read(path)["attempts"]:
            for ref in [attempt["trace"], *attempt.get("stderr_artifacts", [])]:
                if ref:
                    reference(root, Path(ref["path"]), ref["sha256"])


def execute(root, run, config, *, stop_after=None):
    p = config["iteration"]
    initial = {
        "path": config["generation"]["leaf_path"],
        "sha256": config["generation"]["leaf_sha256"],
    }
    bootstrap = {"id": "prefix", "index": 0, **p["bootstrap"]}
    bind(run / "spec.json", bootstrap)
    generated = generate(root, run, run / "data/trajectories", config, initial, bootstrap)
    build(root, run, run / "data", config, bootstrap, generated)
    trained = train(
        run / "data/dataset",
        run / "fit",
        training_config(root, config, initial, bootstrap),
        identity(run, config, run),
        stop_after=stop_after,
    )
    if trained["status"] == "paused_for_resume_check":
        return trained
    if trained["status"] != "complete":
        raise InterruptedError("training stopped with coherent resume checkpoint")
    prefix_candidate = reference(root, run / "fit/trained_candidate.osaval03")
    changed(root / initial["path"], root / prefix_candidate["path"])
    incumbent, actor, learning_start, stagnation = initial, initial, prefix_candidate, 0
    decisions = []
    for index in range(1, p["maximum_generations"] + 1):
        # Minimum three generations always run. One or two further fresh rounds are
        # allowed only with sufficient valid new signal and no repeated deterioration.
        if index > p["minimum_generations"]:
            last = decisions[-1]
            extend = last["new_unique_train"] >= p["extension_minimum_new_labels"] and (
                last["actor_updated"] or last["stagnation"] < 3
            )
            if not extend:
                break
        stopped(run)
        folder = run / "generations" / f"G{index:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        atomic(
            run / "iteration-progress.json",
            encoded(
                {
                    "active_generation": f"G{index:02d}",
                    "completed_generations": index - 1,
                    "actor": actor,
                    "stagnation": stagnation,
                }
            ),
        )
        spec = {
            "id": f"G{index:02d}",
            "index": index,
            "games": p["games_per_generation"],
            "actor": actor,
            "initial": learning_start,
            "stagnation": stagnation,
        }
        bind(folder / "spec.json", spec)
        if (folder / "decision.json").exists():
            decision = read(folder / "decision.json")
            for ref in decision["artifacts"]:
                reference(root, Path(ref["path"]), ref["sha256"])
        else:
            fresh = generate(root, run, folder / "data/trajectories", config, actor, spec)
            data = build(root, run, folder / "data", config, spec, fresh)
            fitting = training_config(root, config, learning_start, spec)
            result = train(
                folder / "data/dataset", folder / "fit", fitting, identity(run, config, folder)
            )
            if result["status"] != "complete":
                raise InterruptedError("training stopped with coherent resume checkpoint")
            candidate = reference(root, folder / "fit/trained_candidate.osaval03")
            difference = changed(root / learning_start["path"], root / candidate["path"])
            audit(root, run, config, root / candidate["path"], folder)
            validation = grouped_arrays(folder / "data/dataset", "development_test")
            groups = {
                name: evaluate_groups(
                    torch_parameters(Phase10VModel.read(root / ref["path"])),
                    validation,
                    config["training"]["batch_size"],
                )
                for name, ref in (("baseline", actor), ("candidate", candidate))
            }
            atomic(folder / "groups.json", encoded(groups))
            arena = compare(
                root, run, config, folder / "arena", candidate, actor, root / p["replay"]["path"]
            )
            decision = {
                **transition(incumbent, actor, candidate, arena, groups, stagnation),
                "generation": spec["id"],
                "tensor_changed_elements": difference,
                "new_unique_train": data["new_unique_positions"]["train"],
                "training_exposures": result["example_exposures"],
                "training_seen": result["seen_unique_positions"],
                "artifacts": [
                    reference(root, folder / name)
                    for name in (
                        "spec.json",
                        "data/dataset/manifest.json",
                        "data/trajectories/result.json",
                        "fit/training.json",
                        "model-audit.json",
                        "arena/arena.json",
                        "groups.json",
                    )
                ],
                "optional_screen": {"status": "not_run", "pass": None},
            }
            atomic(folder / "decision.json", encoded(decision))
        incumbent, actor, stagnation = (
            decision["incumbent"],
            decision["actor"],
            decision["stagnation"],
        )
        learning_start = actor
        decisions.append(decision)
        atomic(
            run / "iteration-progress.json",
            encoded({"completed_generations": index, "actor": actor, "stagnation": stagnation}),
        )
    # Preserve the actor admission chain; changing validation distributions are
    # not a cross-generation ranking. Retain an update even if none is admitted.
    updated = [d for d in decisions if d["actor_updated"]]
    selected = updated[-1]["trained_candidate"] if updated else decisions[-1]["trained_candidate"]
    selection = {
        "candidate": selected,
        "incumbent": incumbent,
        "generations": len(decisions),
        "criterion": "latest admitted candidate, otherwise final distinct comparison candidate",
        "no_strength_claim": True,
    }
    bind(run / "selection.json", selection)
    confirmation = confirmation_starts(root, run, config, initial)
    final = {}
    for name, baseline in p["final_baselines"].items():
        stopped(run)
        final[name] = compare(
            root,
            run,
            config,
            run / "final" / name,
            selected,
            baseline,
            root / p["replay"]["path"],
            confirmation,
        )
    from .evaluator_development import register

    registration = register(root, run, config, root / selected["path"], run / "development")
    report = {
        "status": "complete",
        **selection,
        "decisions": decisions,
        "final_comparison": {name: result["summary"] for name, result in final.items()},
        "development": registration,
        "human_shodan_validated": False,
        "promotion_performed": False,
        "optional_screen_pass": None,
        "novel_confirmation": "available" if confirmation else "missing_unverified",
        "run_sha256": digest(run / "run.json"),
    }
    atomic(run / "c4-result.json", encoded(report))
    return report
