"""C4 fresh trajectories and exact offline labels, with durable itemwise budgets."""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import sqlite3
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from .defense_scenarios import R3Probe, rotate_move, rotate_sfen
from .evaluator_data import MAX_FEATURES, START, Replay, atomic, digest, encoded, symmetry_keys
from .evaluator_data import teacher_config as make_teacher_config
from .evaluator_ledger import DeferredTaskError, Ledger
from .evaluator_training import GROUPS
from .labeling.usi import (
    USIEngine,
    USIIncompleteDepthError,
    USIProcessError,
    USITerminalResult,
    USITimeoutError,
)
from .phase10v_model import sparse_features
from .r4_data import SPLITS, reference, verify_dataset


def read(path: Path):
    return json.loads(path.read_text())


def stopped(run: Path):
    if (run / "STOP").exists():
        raise InterruptedError("requested stop; completed trajectories retained")


def bind(path: Path, value: dict):
    if path.exists() and path.read_bytes() != encoded(value):
        raise ValueError(f"immutable C4 task changed: {path}")
    if not path.exists():
        atomic(path, encoded(value))


class Teacher:
    """One bounded process. Only known local failures become missing labels."""

    def __init__(self, root, generation, ledger, policy):
        self.root, self.ledger, self.policy = root, ledger, policy
        base = make_teacher_config(root, generation)
        self.config = replace(
            base,
            multipv=1,
            timeouts=replace(base.timeouts, search_ms=policy["timeout_ms"]),
        )
        self.engine = None

    def close(self):
        if self.engine is not None:
            self.engine.close()
            self.engine = None

    def observe(self, sfen, identity, *, strong=False):
        depth = self.policy["strong_depth" if strong else "broad_depth"]
        nodes = self.policy["strong_nodes" if strong else "broad_nodes"]
        identity = {
            **identity,
            "sfen": sfen,
            "depth": depth,
            "nodes": nodes,
            "teacher": self.config.binary_sha256,
            "perspective": "side_to_move",
        }
        key, task = self.ledger.task(identity)
        if task["status"] in ("accepted", "deferred"):
            return key, task["result"]
        exits = dict(self.ledger.db.execute("SELECT name,value FROM counters"))
        if exits.get("worker_exits", 0) > self.policy["maximum_worker_exits"]:
            raise RuntimeError("systemic teacher worker failures; preserved task ledger")
        # A crashed attempt is already charged. No retry layer beneath this ledger.
        try:
            self.ledger.begin(key, nodes, maximum=2)
        except DeferredTaskError:
            return key, None
        began = time.monotonic()
        try:
            if self.engine is None:
                self.engine = USIEngine(self.config, self.root)
                self.engine.start()
            result = self.engine.analyze(
                sfen, nodes=nodes, depth=depth, expected_candidates=1, auto_start=False
            )
            if isinstance(result, USITerminalResult):
                value = {"status": "terminal_claim_masked", "outcome": result.outcome}
            else:
                candidate = result.primary
                if candidate.depth != depth or candidate.bound:
                    raise ValueError("accepted teacher label is incomplete or bounded")
                value = {
                    "status": "complete",
                    "candidate": candidate.as_dict(),
                    "elapsed_ms": result.elapsed_ms,
                    "identity": identity,
                }
            self.ledger.finish(
                key, "accepted", result=value, evidence={"elapsed_s": time.monotonic() - began}
            )
            return key, value
        except (USIIncompleteDepthError, USITimeoutError) as error:
            self.ledger.finish(
                key,
                "deferred",
                result={"status": "missing", "reason": type(error).__name__},
                evidence={"elapsed_s": time.monotonic() - began},
            )
            if isinstance(error, USITimeoutError):
                self.close()
            return key, None
        except USIProcessError as error:
            self.close()
            if any(
                token in (str(error) + error.stderr_tail).lower()
                for token in ("std::bad_alloc", "cannot allocate memory", "out of memory")
            ):
                from .evaluator_training import TrainingResourceWaitError

                self.ledger.finish(
                    key,
                    "pending",
                    evidence={
                        "elapsed_s": time.monotonic() - began,
                        "outcome": "allocation_failure",
                    },
                )
                raise TrainingResourceWaitError(
                    "teacher allocation failed; bounded resume"
                ) from error
            with self.ledger.db:
                self.ledger.db.execute("INSERT OR IGNORE INTO counters VALUES('worker_exits',0)")
                self.ledger.db.execute(
                    "UPDATE counters SET value=value+1 WHERE name='worker_exits'"
                )
                exits = self.ledger.db.execute(
                    "SELECT value FROM counters WHERE name='worker_exits'"
                ).fetchone()[0]
            self.ledger.finish(
                key,
                "deferred",
                result={"status": "missing", "reason": type(error).__name__},
                evidence={"elapsed_s": time.monotonic() - began},
            )
            if exits > self.policy["maximum_worker_exits"]:
                raise RuntimeError(
                    "systemic teacher worker failures; preserved task ledger"
                ) from error
            return key, None


def scalar(label):
    return bool(
        label and label.get("status") == "complete" and label["candidate"]["score"]["kind"] == "cp"
    )


def row(base, state, label, key, *, kind="root", pair=None, strong=False):
    if not scalar(label) or state["terminal"] != "None":
        return None
    legal = {c["move"] for c in state["successors"]}
    if label["candidate"]["pv"][0] not in legal:
        raise ValueError("teacher PV head is illegal")
    return {
        **base,
        "sfen": state["sfen"],
        "score": label["candidate"]["score"],
        "score_perspective": "side_to_move",
        "kind": kind,
        "raw_record": pair or key,
        "label_task": key,
        "teacher_depth": label["candidate"]["depth"],
        "teacher_label_origin": "direct_child" if kind == "candidate" else "direct_root",
        "source": "c4_strong" if strong else "c4_broad",
        "round_source": 1,
        "ply": int(state["sfen"].split()[-1]) - 1,
        "history_context": "native full trajectory; teacher SFEN has no repetition history",
    }


def generation_config(config, actor):
    return {
        **config["generation"],
        "leaf_path": actor["path"],
        "leaf_sha256": actor["sha256"],
        "replay_path": config["runtime"]["replay"]["path"],
        "defense_campaign": {
            **config["generation"]["defense_campaign"],
            "probe_path": config["runtime"]["probe"]["path"],
            "probe_sha256": config["runtime"]["probe"]["sha256"],
            "probe_nodes": config["iteration"]["generation"]["nodes"],
        },
    }


def strong_budget(ply, limit):
    phase = 0 if ply < 32 else 1 if ply < 96 else 2
    early, late = min(limit, max(1, limit // 4)), limit // 4
    return phase, (early, limit - early - late, late)[phase]


def generate(root: Path, run: Path, folder: Path, config: dict, actor: dict, spec: dict):
    folder.mkdir(parents=True, exist_ok=True)
    policy = {**config["iteration"]["generation"], **spec.get("generation_overrides", {})}
    generation = generation_config(config, actor)
    bind(
        folder / "plan.json",
        {"actor": actor, "spec": spec, "policy": policy, "run_sha256": digest(run / "run.json")},
    )
    if (folder / "result.json").exists():
        report = read(folder / "result.json")
        for ref in report["artifacts"]:
            reference(root, Path(ref["path"]), ref["sha256"])
        return report
    ledger = Ledger(folder)
    teacher = Teacher(root, generation, ledger, policy["teacher"])
    replay, probe = Replay(root, generation), R3Probe(root, generation)
    opponents = [
        R3Probe(root, generation_config(config, p)) for p in config["iteration"]["opponents"]
    ]
    began, refs, total, outcomes = time.monotonic(), [], Counter(), Counter()
    try:
        for game_id in range(spec["games"]):
            stopped(run)
            path = folder / f"game-{game_id:06d}.json.gz"
            receipt = path.with_suffix(".receipt.json")
            if receipt.exists():
                saved = read(receipt)
                if digest(path) != saved["sha256"]:
                    raise ValueError("committed C4 trajectory changed")
                refs.append(reference(root, path))
                total.update(saved["counts"])
                outcomes[saved["outcome"]] += 1
                continue
            scenarios = policy["starts"]
            scenario = scenarios[game_id % len(scenarios)]
            seed = config["seed"] + spec["index"] * 1_000_003 + game_id
            rotated = bool((game_id // len(scenarios)) % 2)
            initial = rotate_sfen(START) if rotated else START
            prefix = [rotate_move(m) if rotated else m for m in scenario["moves"]]
            game = f"{spec['id']}:{game_id}"
            base = {
                "game": game,
                "family": scenario["family"],
                "group": scenario["group"],
                "split": scenario["split"],
                "generation": spec["id"],
                "seed": seed,
                "actor": actor["sha256"],
                "origin": "new_local_trajectory",
                "scenario_origin": "authored_legal_start_not_user_kifu",
            }
            saved = ledger.load_checkpoint(game_id)
            if saved is None:
                saved = {
                    "moves": prefix,
                    "rows": [],
                    "observations": [],
                    "strong_roots": 0,
                    "strong_phases": [0, 0, 0],
                    "exploration_moves": [],
                    "previous_cp": None,
                }
            state = replay.ask(reset=initial, successors=True)
            for movement in saved["moves"]:
                state = replay.ask(movement=movement, successors=True)
            if saved.get("current_sfen", state["sfen"]) != state["sfen"]:
                raise ValueError("C4 trajectory cursor does not replay exactly")
            for ply in range(len(saved["moves"]), policy["max_plies"]):
                stopped(run)
                if state["terminal"] != "None":
                    break
                moves = saved["moves"]
                current = (
                    probe
                    if game_id % 4 < 2 or ply % 2 == game_id % 2
                    else opponents[(game_id // 4) % len(opponents)]
                )
                predicted = current.search(initial, moves, state["sfen"])
                choice = predicted["best_move"]
                sample = (ply - len(prefix)) % policy["sample_stride"] == 0
                teacher_turn = game_id % 4 == 3 and ply % 2 != game_id % 2
                key, label = (
                    teacher.observe(state["sfen"], {**base, "ply": ply, "kind": "broad"})
                    if sample or teacher_turn
                    else (None, None)
                )
                local = row(base, state, label, key) if label else None
                if sample and local:
                    saved["rows"].append(local)
                if teacher_turn and label and label.get("status") == "complete":
                    choice = label["candidate"]["pv"][0]
                rng = random.Random(seed * 4099 + ply)
                explored = False
                # Root alpha-beta scores nominate moves only. Equal-budget child
                # searches certify the near-good set; nodes are never policy targets.
                if not teacher_turn and ply < 96 and ply % policy["explore_stride"] == 0:
                    nominated = (
                        [r["move"] for r in predicted["iterations"][-1]["roots"][:3]]
                        if predicted["iterations"]
                        else [choice]
                    )
                    candidates = []
                    old_nodes = current.nodes
                    current.nodes = policy["explore_nodes"]
                    try:
                        for move in dict.fromkeys([choice, *nominated]):
                            child = next(c for c in state["successors"] if c["move"] == move)
                            if child["terminal"] == "None":
                                result = current.search(initial, [*moves, move], child["sfen"])
                                if result["depth"] >= 2 and abs(result["score"]) < 20000:
                                    candidates.append((move, -result["score"]))
                        if candidates:
                            best = max(score for _, score in candidates)
                            near = [
                                m
                                for m, score in candidates
                                if score >= best - policy["explore_tolerance_cp"]
                            ]
                            choice = rng.choice(near)
                            explored = choice != predicted["best_move"]
                    finally:
                        current.nodes = old_nodes
                disagreement = (
                    scalar(label) and label["candidate"]["pv"][0] != predicted["best_move"]
                )
                swing = (
                    saved["previous_cp"] is not None
                    and abs(predicted["score"] + saved["previous_cp"]) > 100
                )
                random_stratum = ply % 32 == (game_id % 4) * 8
                phase, allowance = strong_budget(ply, policy["strong_roots_per_game"])
                if (
                    sample
                    and saved["strong_roots"] < policy["strong_roots_per_game"]
                    and saved["strong_phases"][phase] < allowance
                    and (disagreement or swing or random_stratum)
                ):
                    strong_key, strong = teacher.observe(
                        state["sfen"], {**base, "ply": ply, "kind": "strong_root"}, strong=True
                    )
                    local = row(base, state, strong, strong_key, strong=True) if strong else None
                    if local:
                        saved["rows"].append(local)
                    alternatives = [predicted["best_move"]]
                    if strong and strong.get("status") == "complete":
                        alternatives.append(strong["candidate"]["pv"][0])
                    if len(set(alternatives)) < 2 and predicted["iterations"]:
                        alternative = next(
                            (
                                r["move"]
                                for r in predicted["iterations"][-1]["roots"]
                                if r["move"] not in alternatives
                            ),
                            None,
                        )
                        if alternative:
                            alternatives.append(alternative)
                    pair = f"{game}:{ply}:direct_children"
                    for alternative in dict.fromkeys(alternatives):
                        child = replay.ask(movement=alternative, successors=True)
                        child_key, child_label = teacher.observe(
                            child["sfen"],
                            {**base, "ply": ply, "kind": "child", "move": alternative},
                            strong=True,
                        )
                        local = row(
                            base,
                            child,
                            child_label,
                            child_key,
                            kind="candidate",
                            pair=pair,
                            strong=True,
                        )
                        if local:
                            saved["rows"].append(local)
                        # Learn the position after a strong reply and an attack/retreat
                        # continuation, with fresh labels at each changed position.
                        for continuation in range(policy["reply_plies"]):
                            if (
                                not child_label
                                or child_label.get("status") != "complete"
                                or child["terminal"] != "None"
                            ):
                                break
                            reply = child_label["candidate"]["pv"][0]
                            child = replay.ask(movement=reply, successors=True)
                            if child["terminal"] != "None":
                                break
                            child_key, child_label = teacher.observe(
                                child["sfen"],
                                {
                                    **base,
                                    "ply": ply,
                                    "kind": "reply",
                                    "move": alternative,
                                    "continuation": continuation,
                                },
                                strong=True,
                            )
                            local = row(base, child, child_label, child_key, strong=True)
                            if local:
                                saved["rows"].append(local)
                        state = replay.ask(reset=initial, successors=True)
                        for movement in moves:
                            state = replay.ask(movement=movement, successors=True)
                    saved["strong_roots"] += 1
                    saved["strong_phases"][phase] += 1
                saved["observations"].append(
                    {
                        "sfen": state["sfen"],
                        "move": choice,
                        "actor_move": predicted["best_move"],
                        "cp": predicted["score"],
                        "depth": predicted["depth"],
                        "nodes": predicted["nodes"],
                        "teacher_task": key,
                        "teacher_turn": teacher_turn,
                        "teacher_move_used": bool(
                            teacher_turn and label and label.get("status") == "complete"
                        ),
                        "strong_reason": {
                            "disagreement": bool(disagreement),
                            "swing": swing,
                            "random": random_stratum,
                        },
                    }
                )
                if explored:
                    saved["exploration_moves"].append(ply)
                state = replay.ask(movement=choice, successors=True)
                saved["moves"] = [*moves, choice]
                saved["previous_cp"] = predicted["score"]
                saved["current_sfen"] = state["sfen"]
                ledger.checkpoint(game_id, saved)
            outcome = state["terminal"] if state["terminal"] != "None" else "max_plies_unscored"
            content = {
                **base,
                **saved,
                "initial_sfen": initial,
                "outcome": outcome,
                "opponent_type": "selfplay"
                if game_id % 4 < 2
                else "historical"
                if game_id % 4 == 2
                else "offline_teacher",
                "opponent": config["iteration"]["opponents"][(game_id // 4) % len(opponents)],
                "outcome_used_as_target": False,
            }
            # A crash between shard and receipt leaves the complete durable cursor.
            atomic(path, gzip.compress(encoded(content), mtime=0))
            counts = {
                "games": 1,
                "generated_boards": len(saved["observations"]),
                "label_rows": len(saved["rows"]),
                "strong_roots": saved["strong_roots"],
                "exploration_moves": len(saved["exploration_moves"]),
            }
            atomic(receipt, encoded({"sha256": digest(path), "counts": counts, "outcome": outcome}))
            refs.append(reference(root, path))
            total.update(counts)
            outcomes[outcome] += 1
            atomic(folder / "progress.json", encoded(dict(total)))
            print(json.dumps({"generation": spec["id"], **dict(total)}), flush=True)
        report = {
            "status": "complete",
            **dict(total),
            "artifacts": refs,
            "outcomes": dict(outcomes),
            "elapsed_this_invocation_seconds": time.monotonic() - began,
            "teacher_tasks": dict(
                ledger.db.execute("SELECT status,count(*) FROM tasks GROUP BY status")
            ),
            "worker_counters": dict(ledger.db.execute("SELECT name,value FROM counters")),
        }
        atomic(folder / "result.json", encoded(report))
        return report
    finally:
        teacher.close()
        ledger.close()
        replay.close()
        probe.close()
        for opponent in opponents:
            opponent.close()


def build(root: Path, run: Path, folder: Path, config: dict, spec: dict, generated: dict):
    """Stream rows through a disk index; source trajectories keep their original split."""
    output = folder / "dataset"
    if output.exists():
        return verify_dataset(root, output)
    replay_ref = config["iteration"]["replay"]
    parent = root / replay_ref["path"]
    verify_dataset(root, parent, replay_ref["manifest_sha256"])
    index_path = run / "position-index.sqlite3"
    db = sqlite3.connect(index_path)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS owners(
          key TEXT PRIMARY KEY,split TEXT NOT NULL,generation TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
    """)
    if not db.execute("SELECT value FROM meta WHERE key='base'").fetchone():
        with db:
            for split in SPLITS:
                with gzip.open(parent / f"{split}-rows.jsonl.gz", "rt") as f:
                    for line in f:
                        r = json.loads(line)
                        db.execute(
                            "INSERT OR IGNORE INTO owners VALUES(?,?, 'ancestor')",
                            (r["symmetry_key"], split),
                        )
            # These are hash-only guards; no final-holdout position is opened.
            guard = read(root / config["generation"]["split_guard_path"])
            for values in guard.values():
                if isinstance(values, list):
                    for r in values:
                        if isinstance(r, dict) and "position_sha256" in r:
                            db.execute(
                                "INSERT OR REPLACE INTO owners VALUES(?,'blocked','guard')",
                                (r["position_sha256"],),
                            )
            for key in read(root / config["generation"]["development_exclusions_path"])[
                "symmetry_keys"
            ]:
                db.execute("INSERT OR REPLACE INTO owners VALUES(?,'blocked','guard')", (key,))
            db.execute("INSERT INTO meta VALUES('base',?)", (replay_ref["manifest_sha256"],))
    if (
        db.execute("SELECT value FROM meta WHERE key='base'").fetchone()[0]
        != replay_ref["manifest_sha256"]
    ):
        raise ValueError("C4 ancestry index belongs to another replay")
    staging = folder / "dataset-building"
    staging.mkdir(parents=True, exist_ok=True)
    # This staging database is a reproducible materialization, never a task ledger.
    rows = sqlite3.connect(staging / "rows.sqlite3")
    rows.execute(
        "CREATE TABLE IF NOT EXISTS rows(key TEXT PRIMARY KEY,split TEXT,source TEXT,payload TEXT)"
    )
    rows.execute("DELETE FROM rows")
    excluded = Counter()
    with db, rows:
        for ref in generated["artifacts"]:
            path = root / ref["path"]
            reference(root, path, ref["sha256"])
            raw = json.loads(gzip.decompress(path.read_bytes()))
            # Unlabelled observations also count as previously encountered positions.
            # A later successful reanalysis is not called a newly generated board.
            for observation in raw["observations"]:
                key = min(symmetry_keys(observation["sfen"]))
                db.execute(
                    "INSERT OR IGNORE INTO owners VALUES(?,?,?)", (key, raw["split"], spec["id"])
                )
            for r in raw["rows"]:
                keys = symmetry_keys(r["sfen"])
                key = min(keys)
                placeholders = ",".join("?" for _ in keys)
                owners = db.execute(
                    f"SELECT split,generation FROM owners WHERE key IN ({placeholders})",
                    tuple(keys),
                ).fetchall()
                if any(s != r["split"] for s, _ in owners):
                    excluded["split_or_guard_conflict"] += 1
                    continue
                r.update(
                    symmetry_key=key,
                    new_to_known_ancestry=not owners or all(g == spec["id"] for _, g in owners),
                )
                db.execute(
                    "INSERT OR IGNORE INTO owners VALUES(?,?,?)", (key, r["split"], spec["id"])
                )
                existing = rows.execute("SELECT source FROM rows WHERE key=?", (key,)).fetchone()
                if existing and (existing[0] == "c4_strong" or r["source"] != "c4_strong"):
                    excluded["within_generation_duplicate"] += 1
                    continue
                rows.execute(
                    "INSERT OR REPLACE INTO rows VALUES(?,?,?,?)",
                    (key, r["split"], r["source"], encoded(r).decode()),
                )
    # Select bounded historical replay by hash, not by taking the file's prefix.
    budget = spec.get("replay_rows", config["iteration"]["replay_rows"])
    replay_rows = {s: [] for s in SPLITS}
    import heapq

    heaps = {g: [] for g in GROUPS}
    for split in SPLITS:
        with gzip.open(parent / f"{split}-rows.jsonl.gz", "rt") as f:
            for line in f:
                r = json.loads(line)
                if split != "train":
                    replay_rows[split].append(r)
                    continue
                priority = int(
                    hashlib.sha256(
                        f"{config['seed']}:{spec['index']}:{r['symmetry_key']}".encode()
                    ).hexdigest(),
                    16,
                )
                bucket = heaps[r["group"]]
                item = (-priority, r["symmetry_key"], r)
                if len(bucket) < budget // 4:
                    heapq.heappush(bucket, item)
                elif item[:2] > bucket[0][:2]:
                    heapq.heapreplace(bucket, item)
    replay_rows["train"] = [r for bucket in heaps.values() for _, _, r in bucket]
    refs = [reference(root, parent / "manifest.json"), *generated["artifacts"]]
    past = []
    for previous in sorted((run / "generations").glob("G*/data/dataset/train-rows.jsonl.gz")):
        if previous.parent == output:
            continue
        with gzip.open(previous, "rt") as f:
            for line in f:
                r = json.loads(line)
                if r["source"] not in ("c4_broad", "c4_strong"):
                    continue
                priority = int(hashlib.sha256(r["symmetry_key"].encode()).hexdigest(), 16)
                item = (-priority, r["symmetry_key"], encoded(r))
                if len(past) < config["iteration"]["past_rows"]:
                    heapq.heappush(past, item)
                elif item[:2] > past[0][:2]:
                    heapq.heapreplace(past, item)
        refs.append(reference(root, previous))
    replay_rows["train"].extend(
        {**json.loads(payload), "source": "c4_past", "round_source": 0} for _, _, payload in past
    )
    counts, sources, groups, new_unique, pairs, games = {}, {}, {}, {}, {}, {}
    for split in SPLITS:
        selected = {}
        for r in replay_rows[split]:
            if (
                db.execute("SELECT split FROM owners WHERE key=?", (r["symmetry_key"],)).fetchone()[
                    0
                ]
                == split
            ):
                selected[r["symmetry_key"]] = {**r, "round_source": 0}
        for (payload,) in rows.execute(
            "SELECT payload FROM rows WHERE split=? ORDER BY key", (split,)
        ):
            r = json.loads(payload)
            selected[r["symmetry_key"]] = r
        values = list(selected.values())
        n = len(values)
        if not n:
            raise ValueError(f"empty C4 {split}; preserve independent generation evidence")
        arrays = {
            "features": np.zeros((n, 2, MAX_FEATURES), dtype=np.int32),
            "lengths": np.zeros((n, 2), dtype=np.int64),
            "targets": np.zeros(n, dtype=np.float32),
            "groups": np.zeros(n, dtype=np.uint8),
            "sources": np.zeros(n, dtype=np.uint8),
            "origins": np.zeros(n, dtype=np.uint8),
            "novelty": np.zeros(n, dtype=np.uint8),
            "sequences": np.zeros(n, dtype=np.int32),
            "partners": np.full(n, -1, dtype=np.int32),
        }
        sequences, siblings, source_counts, group_counts = {}, {}, Counter(), Counter()
        with gzip.open(staging / f"{split}-rows.jsonl.gz", "wb") as stream:
            for i, r in enumerate(values):
                own, other, stm = sparse_features(r["sfen"])
                for side, features in enumerate((other, own) if stm else (own, other)):
                    if not 0 < len(features) <= MAX_FEATURES:
                        raise ValueError("invalid sparse C4 input")
                    arrays["features"][i, side, : len(features)] = features
                    arrays["lengths"][i, side] = len(features)
                arrays["targets"][i] = np.clip(r["score"]["value"], -20000, 20000)
                arrays["groups"][i] = GROUPS.index(r["group"])
                arrays["sources"][i] = r.get("round_source", 0)
                origin = (
                    5
                    if r["source"] == "c4_strong"
                    else 4
                    if r["source"] == "c4_broad"
                    else 6
                    if r["source"] == "c4_past"
                    else {
                        "generated": 0,
                        "r3_replay": 1,
                        "nodchip_hao_depth9": 2,
                        "c3_apery_counterfactual": 3,
                    }[r["source"]]
                )
                arrays["origins"][i] = origin
                arrays["novelty"][i] = (
                    (2 if r.get("new_to_known_ancestry") else 1)
                    if r.get("generation") == spec["id"]
                    else 0
                )
                sequence = str(r["game"])
                arrays["sequences"][i] = sequences.setdefault(sequence, len(sequences))
                if r["kind"] == "candidate" and r.get("raw_record"):
                    siblings.setdefault((r["source"], sequence, str(r["raw_record"])), []).append(i)
                games[sequence] = {k: r[k] for k in ("game", "family", "group", "split")}
                source_counts[r["source"]] += 1
                group_counts[r["group"]] += 1
                stream.write(encoded(r) + b"\n")
        for indexes in siblings.values():
            if len(indexes) >= 2:
                a = min(indexes, key=lambda i: arrays["targets"][i])
                b = max(indexes, key=lambda i: arrays["targets"][i])
                if a != b and abs(float(arrays["targets"][a] - arrays["targets"][b])) > 50:
                    arrays["partners"][a] = b
                    arrays["partners"][b] = a
        for name, a in arrays.items():
            np.save(staging / f"{split}-{name}.npy", a)
        pairs[split] = int((arrays["partners"] >= 0).sum()) // 2
        counts[split], sources[split], groups[split] = n, dict(source_counts), dict(group_counts)
        new_unique[split] = sum(
            r.get("generation") == spec["id"] and r.get("new_to_known_ancestry", False)
            for r in values
        )
    rows.close()
    db.close()
    # The building index is regenerable and is not an input to learning.
    (staging / "rows.sqlite3").unlink()
    report = {
        "schema": "open_shogiai_r4_data/v1",
        "generation": spec["id"],
        "unique_positions": counts,
        "new_unique_positions": new_unique,
        "source_rows": sources,
        "origin_names": {
            "0": "Apery generated replay",
            "1": "r3 replay",
            "2": "Hao depth9 replay",
            "3": "C3 counterfactual replay",
            "4": "C4 broad",
            "5": "C4 strong",
            "6": "C4 past generations",
        },
        "distributions": groups,
        "candidate_pairs": pairs,
        "source_games": list(games.values()),
        "excluded": dict(excluded),
        "inputs": refs,
        "sealed_holdout_opened": False,
        "ancestral_exposure": "known manifests deduplicated; older unrecorded history unknown",
        "artifacts": [
            {"path": p.name, "sha256": digest(p), "bytes": p.stat().st_size}
            for p in sorted(staging.iterdir())
            if p.is_file()
        ],
    }
    atomic(staging / "manifest.json", encoded(report))
    staging.rename(output)
    return report
