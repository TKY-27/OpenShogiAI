"""C3 offline local counterfactuals and sibling comparisons; never imported by play."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .defense_scenarios import R3Probe, rotate_move, rotate_sfen
from .evaluator_data import (
    MAX_FEATURES,
    START,
    Replay,
    atomic,
    digest,
    encoded,
    symmetry_keys,
    teacher_config,
)
from .evaluator_training import GROUPS
from .labeling.usi import USIEngine, USIIncompleteDepthError, USITerminalResult, USITimeoutError
from .phase10v_model import sparse_features
from .r4_data import SPLITS, reference, verify_dataset


def generate(root: Path, config: dict, output: Path) -> dict:
    plan = config["c3_preparation"]
    output.mkdir(parents=True, exist_ok=True)
    generation = {
        **config["generation"],
        "replay_path": "target/release/examples/position_audit",
        "probe_path": "target/release/examples/core_probe",
    }
    generation["defense_campaign"] = {
        **generation["defense_campaign"],
        "probe_path": generation["probe_path"],
        "probe_sha256": digest(root / generation["probe_path"]),
        "probe_nodes": plan["probe_nodes"],
    }
    # Dataset assembly follows generation and cannot change a cached teacher task.
    bound_generation = {k: v for k, v in generation.items() if k != "prepared_dataset"}
    binding = {"plan": plan, "generation": bound_generation}
    if (output / "plan.json").exists():
        previous = json.loads((output / "plan.json").read_text())
        previous["generation"].pop("prepared_dataset", None)
        if previous != binding:
            raise ValueError("C3 generation plan changed")
    else:
        atomic(output / "plan.json", encoded(binding))
    replay = Replay(root, generation)
    probe = R3Probe(root, generation)
    try:
        with USIEngine(teacher_config(root, generation), root) as teacher:

            def observe(state, base, kind, turn_moves):
                key = digest_bytes(encoded([base, kind, state["sfen"]]))
                path = output / (key + ".json")
                if path.exists():
                    return json.loads(path.read_text())
                result = {
                    **base,
                    "kind": kind,
                    "sfen": state["sfen"],
                    "moves": turn_moves,
                    "history_context": "native prefix; teacher SFEN omits repetition",
                    "rows": [],
                    "status": "missing",
                }
                if state["terminal"] != "None":
                    result["status"] = "terminal_excluded"
                else:
                    try:
                        analysis = teacher.analyze(
                            state["sfen"],
                            depth=plan["depth"],
                            nodes=plan["nodes"],
                            expected_candidates=min(3, len(state["successors"])),
                        )
                        if isinstance(analysis, USITerminalResult):
                            result["status"] = "terminal_claim_excluded"
                        else:
                            result.update(
                                status="complete",
                                candidates=[c.as_dict() for c in analysis.candidates],
                            )
                            children = {c["move"]: c for c in state["successors"]}
                            for c in analysis.candidates:
                                if c.pv[0] not in children or c.bound or c.depth != plan["depth"]:
                                    raise ValueError("invalid completed candidate")

                            def row(sfen, score, row_kind, move=None):
                                return {
                                    **base,
                                    "sfen": sfen,
                                    "score": score,
                                    "score_perspective": "side_to_move",
                                    "kind": row_kind,
                                    "move": move,
                                    "source": "c3_apery_counterfactual",
                                    "raw_record": key,
                                    "ply": int(sfen.split()[-1]) - 1,
                                    "round_source": 0,
                                    "teacher_requested_root_depth": plan["depth"],
                                    "teacher_label_origin": "root_multipv_child_negated"
                                    if row_kind == "candidate"
                                    else "direct_root",
                                    "history_context": result["history_context"],
                                }

                            first = analysis.primary
                            if first.score.kind == "cp":
                                result["rows"].append(
                                    row(state["sfen"], first.score.as_dict(), "root")
                                )
                            for c in analysis.candidates:
                                child = children[c.pv[0]]
                                if c.score.kind == "cp" and child["terminal"] == "None":
                                    result["rows"].append(
                                        row(
                                            child["sfen"],
                                            {"kind": "cp", "value": -c.score.value},
                                            "candidate",
                                            c.pv[0],
                                        )
                                    )
                    except (USIIncompleteDepthError, USITimeoutError) as error:
                        result.update(
                            reason=type(error).__name__, evidence=teacher.search_diagnostics
                        )
                atomic(path, encoded(result))
                return result

            for scenario in plan["scenarios"]:
                for rotated in (False, True):
                    game = f"c3:{scenario['id']}:{int(rotated)}"
                    initial = rotate_sfen(START) if rotated else START
                    moves = [rotate_move(m) if rotated else m for m in scenario["moves"]]
                    state = replay.ask(reset=initial, successors=True)
                    for move in moves:
                        if move not in {c["move"] for c in state["successors"]}:
                            raise ValueError(f"illegal scenario {game}: {move}")
                        state = replay.ask(movement=move, successors=True)
                    for index in range(plan["roots_per_scenario"]):
                        base = {
                            "game": game,
                            "family": "c3:" + scenario["id"],
                            "group": scenario["group"],
                            "split": scenario["split"],
                            "scenario_origin": "authored_legal_example_not_user_kifu",
                        }
                        observed = observe(state, base, "trajectory", moves)
                        if observed["status"] != "complete":
                            break  # Missing one trajectory never blocks the other scenarios.
                        candidates = observed["candidates"]
                        if index % 3 == 0 and scenario["split"] == "train":
                            prediction = probe.search(initial, moves, state["sfen"])
                            choice = prediction["best_move"]
                            if choice:
                                wrong = replay.ask(movement=choice, successors=True)
                                # Bind the model candidate to strong replies and recovery.
                                observe(wrong, base, "model_candidate", [*moves, choice])
                                replay.ask(reset=initial)
                                for move in moves:
                                    replay.ask(movement=move)
                        chosen = candidates[
                            (index // 3) % len(candidates) if scenario["split"] == "train" else 0
                        ]["pv"][0]
                        moves.append(chosen)
                        state = replay.ask(movement=chosen, successors=True)
                        if state["terminal"] != "None":
                            break
                    print(json.dumps({"scenario": game, "last_ply": len(moves)}), flush=True)
    finally:
        replay.close()
        probe.close()
    files = sorted(p for p in output.glob("*.json") if p.name not in {"plan.json", "result.json"})
    results = [json.loads(p.read_text()) for p in files]
    report = {
        "status": "complete",
        "observations": len(results),
        "states": dict(Counter(r["status"] for r in results)),
        "rows": sum(len(r["rows"]) for r in results),
        "inputs": [reference(root, p) for p in files],
    }
    atomic(output / "result.json", encoded(report))
    return report


def digest_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def build(root: Path, config: dict, output: Path) -> dict:
    if output.exists():
        return verify_dataset(root, output)
    plan = config["c3_preparation"]
    parent = root / plan["replay"]["path"]
    old = verify_dataset(root, parent, plan["replay"]["manifest_sha256"])
    focus = root / plan["focus"]
    prepared = json.loads((focus / "result.json").read_text())
    refs = [
        reference(root, parent / "manifest.json", plan["replay"]["manifest_sha256"]),
        reference(root, focus / "result.json"),
    ]
    refs.extend(reference(root, parent / r["path"], r["sha256"]) for r in old["artifacts"])
    incoming = []
    for ref in prepared["inputs"]:
        refs.append(reference(root, Path(ref["path"]), ref["sha256"]))
        incoming.extend(json.loads((root / ref["path"]).read_text())["rows"])
    owners = {}
    for split in SPLITS:
        with gzip.open(parent / f"{split}-rows.jsonl.gz", "rt") as f:
            for line in f:
                row = json.loads(line)
                owners[row["symmetry_key"]] = split
    # The entire historical eval ledger (not merely the C2 subsample) stays protected.
    protected = root / config["generation"]["defense_campaign"]["replay_dataset"]["path"]
    for split in ("validation", "development_test"):
        path = protected / f"{split}-rows.jsonl.gz"
        refs.append(reference(root, path))
        with gzip.open(path, "rt") as f:
            for line in f:
                row = json.loads(line)
                owners[row["symmetry_key"]] = split
    guard = root / config["generation"]["split_guard_path"]
    refs.append(reference(root, guard))
    blocked = {
        r["position_sha256"]
        for rows in json.loads(guard.read_text()).values()
        if isinstance(rows, list)
        for r in rows
        if isinstance(r, dict) and "position_sha256" in r
    }
    exclusions = root / config["generation"]["development_exclusions_path"]
    refs.append(reference(root, exclusions))
    blocked.update(json.loads(exclusions.read_text())["symmetry_keys"])
    conflicts = set()
    for row in incoming:
        keys = symmetry_keys(row["sfen"])
        key = row["symmetry_key"] = min(keys)
        if keys & blocked or (key in owners and owners[key] != row["split"]):
            conflicts.add(key)
        owners.setdefault(key, row["split"])
    used = set()
    new = {s: [] for s in SPLITS}
    # Reject duplicates, including overlaps with old train, preserving immutable labels.
    for split in SPLITS:
        with gzip.open(parent / f"{split}-rows.jsonl.gz", "rt") as f:
            used.update(json.loads(line)["symmetry_key"] for line in f)
    for row in incoming:
        key = row["symmetry_key"]
        if key in used or key in conflicts:
            continue
        used.add(key)
        new[row["split"]].append(row)
    staging = output.with_name(output.name + "-building")
    staging.mkdir(parents=True)
    pair_counts = {}
    source_rows = {}
    sizes = {}
    distributions = {}
    new_games = {}
    for split in SPLITS:
        old_arrays = {
            name: np.load(parent / f"{split}-{name}.npy", mmap_mode="r")
            for name in (
                "features",
                "lengths",
                "targets",
                "groups",
                "sources",
                "origins",
                "sequences",
            )
        }
        n = len(old_arrays["targets"])
        size = n + len(new[split])
        sizes[split] = size
        arrays = {
            name: np.lib.format.open_memmap(
                staging / f"{split}-{name}.npy",
                mode="w+",
                dtype=a.dtype,
                shape=(size, *a.shape[1:]),
            )
            for name, a in old_arrays.items()
        }
        for name, a in old_arrays.items():
            arrays[name][:n] = a
        next_sequence = int(old_arrays["sequences"].max()) + 1
        sequences = {}
        siblings = {}
        origins = Counter()
        groups = Counter()

        def record(row, index, siblings=siblings, origins=origins, groups=groups):
            if row["kind"] == "candidate" and "raw_record" in row:
                siblings.setdefault(
                    (row["source"], str(row["game"]), str(row["raw_record"])), []
                ).append(index)
            origins[row["source"]] += 1
            groups[row["group"]] += 1

        with gzip.open(staging / f"{split}-rows.jsonl.gz", "wb") as dst:
            with gzip.open(parent / f"{split}-rows.jsonl.gz", "rt") as f:
                for i, line in enumerate(f):
                    row = json.loads(line)
                    record(row, i)
                    dst.write(encoded(row) + b"\n")
            for i, row in enumerate(new[split], n):
                own, other, stm = sparse_features(row["sfen"])
                for side, features in enumerate((other, own) if stm else (own, other)):
                    if not 0 < len(features) <= MAX_FEATURES:
                        raise ValueError("invalid feature count")
                    arrays["features"][i, side, : len(features)] = features
                    arrays["lengths"][i, side] = len(features)
                arrays["targets"][i] = np.clip(row["score"]["value"], -20000, 20000)
                arrays["groups"][i] = GROUPS.index(row["group"])
                arrays["sources"][i] = 0
                arrays["origins"][i] = 3
                arrays["sequences"][i] = sequences.setdefault(
                    row["game"], next_sequence + len(sequences)
                )
                record(row, i)
                dst.write(encoded(row) + b"\n")
                new_games[row["game"]] = {k: row[k] for k in ("game", "family", "group", "split")}
        partners = np.full(size, -1, dtype=np.int32)
        for indexes in siblings.values():
            if len(indexes) < 2:
                continue
            a = min(indexes, key=lambda i: arrays["targets"][i])
            b = max(indexes, key=lambda i: arrays["targets"][i])
            if a != b and abs(float(arrays["targets"][a] - arrays["targets"][b])) > 50:
                partners[a] = b
                partners[b] = a
        np.save(staging / f"{split}-partners.npy", partners)
        pair_counts[split] = int((partners >= 0).sum()) // 2
        source_rows[split] = dict(origins)
        distributions[split] = dict(groups)
        for a in arrays.values():
            a.flush()
    report = {
        **old,
        "plan": plan,
        "inputs": refs,
        "artifacts": [
            {"path": p.name, "sha256": digest(p), "bytes": p.stat().st_size}
            for p in sorted(staging.iterdir())
        ],
        "unique_positions": sizes,
        "source_rows": source_rows,
        "distributions": distributions,
        "source_games": old["source_games"] + list(new_games.values()),
        "candidate_pairs": pair_counts,
        "added_counterfactual_rows": {s: len(v) for s, v in new.items()},
        "new_conflicts": len(conflicts),
        "origins": {**old["origins"], "3": "c3_apery_counterfactual"},
        "objective": "sibling ordering + scalar replay; tolerance50cp; margin cap600cp",
        "sealed_holdout_opened": False,
    }
    atomic(staging / "manifest.json", encoded(report))
    staging.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config = json.loads(args.config.read_text())
    generate(root, config, root / config["c3_preparation"]["focus"])
    if not args.generate_only:
        report = build(root, config, root / config["c3_preparation"]["output"])
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in ("unique_positions", "candidate_pairs", "added_counterfactual_rows")
                }
            )
        )


if __name__ == "__main__":
    main()
