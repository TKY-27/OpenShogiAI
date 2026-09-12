"""Finite local teacher task ledger; transactions survive worker/process interruption."""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import random
import sqlite3
import time
from pathlib import Path


class DeferredTaskError(RuntimeError):
    """A label is unavailable; its trajectory checkpoint remains resumable."""


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class Ledger:
    def __init__(self, output: Path):
        output.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(output / "tasks.sqlite3", timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, identity TEXT NOT NULL,
          status TEXT NOT NULL, attempts TEXT NOT NULL, result TEXT, updated REAL NOT NULL);
        CREATE INDEX IF NOT EXISTS tasks_updated ON tasks(updated);
        CREATE TABLE IF NOT EXISTS counters(name TEXT PRIMARY KEY, value INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS checkpoints(game INTEGER PRIMARY KEY, payload TEXT NOT NULL);
        """)

    def close(self):
        self.db.close()

    def task(self, identity):
        payload = _json(identity)
        key = hashlib.sha256(payload.encode()).hexdigest()
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO tasks VALUES(?,?, 'pending','[]',NULL,?)",
                (key, payload, time.time()),
            )
        return key, self.get(key)

    def get(self, key):
        row = self.db.execute(
            "SELECT status,attempts,result FROM tasks WHERE id=?", (key,)
        ).fetchone()
        return {
            "status": row[0],
            "attempts": json.loads(row[1]),
            "result": json.loads(row[2]) if row[2] else None,
        }

    def begin(self, key, nodes, maximum=2, policy=None):
        with self.db:
            value = self.get(key)
            if len(value["attempts"]) >= maximum:
                self.finish(key, "deferred")
                raise DeferredTaskError("cumulative teacher attempt budget exhausted")
            if policy and "maximum_task_seconds" in policy:
                consumed = sum(
                    a.get("elapsed_s", policy["maximum_attempt_seconds"]) for a in value["attempts"]
                )
                if consumed + policy["maximum_attempt_seconds"] > policy["maximum_task_seconds"]:
                    self.db.execute("UPDATE tasks SET status='deferred' WHERE id=?", (key,))
                    self.db.commit()
                    raise DeferredTaskError("task wall budget exhausted")
            reservation = None
            if policy and nodes == policy.get("hard_nodes"):
                for name, initial in (
                    ("hard_attempts", policy.get("inherited_hard_attempts", 0)),
                    ("hard_seconds", policy.get("inherited_hard_seconds", 0)),
                ):
                    self.db.execute("INSERT OR IGNORE INTO counters VALUES(?,?)", (name, initial))
                counters = dict(self.db.execute("SELECT name,value FROM counters"))
                reservation = policy["maximum_attempt_seconds"]
                if (
                    counters["hard_attempts"] >= policy["maximum_hard_attempts"]
                    or counters["hard_seconds"] + reservation > policy["maximum_hard_seconds"]
                ):
                    self.db.execute("UPDATE tasks SET status='deferred' WHERE id=?", (key,))
                    self.db.commit()
                    raise DeferredTaskError("hard queue budget exhausted")
                self.db.execute("UPDATE counters SET value=value+1 WHERE name='hard_attempts'")
                self.db.execute(
                    "UPDATE counters SET value=value+? WHERE name='hard_seconds'", (reservation,)
                )
            attempts = value["attempts"] + [
                {
                    "nodes": nodes,
                    "started": time.time(),
                    "outcome": "interrupted",
                    "reserved_seconds": reservation,
                }
            ]
            self.db.execute(
                "UPDATE tasks SET attempts=?,status='running',updated=? WHERE id=?",
                (_json(attempts), time.time(), key),
            )

    def finish(self, key, status, *, result=None, evidence=None):
        with self.db:
            value = self.get(key)
            attempts = value["attempts"]
            if evidence and attempts:
                reserved = attempts[-1].get("reserved_seconds")
                if reserved is not None and "elapsed_s" in evidence:
                    self.db.execute(
                        "UPDATE counters SET value=value+? WHERE name='hard_seconds'",
                        (evidence["elapsed_s"] - reserved,),
                    )
                    attempts[-1]["reserved_seconds"] = None
                attempts[-1].update(evidence)
            self.db.execute(
                "UPDATE tasks SET status=?,attempts=?,result=?,updated=? WHERE id=?",
                (
                    status,
                    _json(attempts),
                    _json(result) if result is not None else None,
                    time.time(),
                    key,
                ),
            )

    def checkpoint(self, game, payload):
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO checkpoints VALUES(?,?)", (game, _json(payload))
            )

    def load_checkpoint(self, game):
        row = self.db.execute("SELECT payload FROM checkpoints WHERE game=?", (game,)).fetchone()
        return json.loads(row[0]) if row else None

    def increment(self, name):
        with self.db:
            self.db.execute(
                "INSERT INTO counters VALUES(?,1) ON CONFLICT(name) DO UPDATE SET value=value+1",
                (name,),
            )
            return self.db.execute("SELECT value FROM counters WHERE name=?", (name,)).fetchone()[0]

    def check_health(self):
        # Repeated failures at one SFEN cannot trip a distinct-task outage gate.
        recent = list(
            self.db.execute(
                "SELECT identity,status FROM tasks WHERE status != 'pending' "
                "ORDER BY updated DESC LIMIT 32"
            )
        )
        games = {json.loads(row[0])["game"] for row in recent}
        if (
            len(recent) >= 32
            and len(games) >= 8
            and sum(row[1] != "accepted" for row in recent) >= 24
        ):
            raise RuntimeError(
                "teacher output health gate: at least 24 of 32 distinct tasks "
                "failed across eight trajectories"
            )

    def summary(self):
        result = dict(self.db.execute("SELECT status,count(*) FROM tasks GROUP BY status"))
        result["retried"] = sum(
            len(json.loads(r[0])) > 1 for r in self.db.execute("SELECT attempts FROM tasks")
        )
        result["last_valid_output_at"] = self.db.execute(
            "SELECT max(updated) FROM tasks WHERE status='accepted'"
        ).fetchone()[0]
        return result


def task_identity(config, game, ply, branch, state, moves):
    family = None
    split = None
    if config.get("defense_campaign"):
        from .defense_scenarios import assignment

        family, _ = assignment(config, game)
        split = family["split"]
    return {
        "game": game,
        "ply": ply,
        "branch": branch,
        "sfen": state["sfen"],
        "history": moves,
        "split": split,
        "group": family["group"] if family else None,
        "teacher": config["teacher_binary_sha256"],
        "teacher_config": config["teacher_config_sha256"],
        "depth": config.get("teacher_depth"),
        "acceptance": "exact_complete_multipv_v2",
    }


def pack_result(result):
    return {"type": type(result).__name__, "value": dataclasses.asdict(result)}


def unpack_result(value):
    from .labeling.usi import USICandidate, USIScore, USISearchResult, USITerminalResult

    data = value["value"]
    if value["type"] == "USITerminalResult":
        return USITerminalResult(**data)
    candidates = tuple(
        USICandidate(**{**c, "score": USIScore(**c["score"]), "pv": tuple(c["pv"])})
        for c in data["candidates"]
    )
    return USISearchResult(**{**data, "candidates": candidates})


def restore_rng(value):
    return tuple(restore_rng(v) if isinstance(v, list) else v for v in value)


def import_failure(output: Path, config: dict, failure_paths: list[Path], initial_sfen: str):
    """Import immutable old evidence; never infer a completed trajectory from its prefix."""
    pairs = [(path, json.loads(path.read_text())) for path in failure_paths]
    parent = json.loads((failure_paths[0].parents[1] / "generation.json").read_text())
    known = {}
    for nodes in (config["teacher_nodes"], config["teacher_retry_nodes"]):
        request = (
            parent
            if nodes == config["teacher_nodes"]
            else {**parent, "teacher_nodes": nodes, "teacher_retry_nodes": None}
        )
        known[hashlib.sha256(_json(request).encode()).hexdigest()] = nodes
    if any(f["generation_config_sha256"] not in known for _, f in pairs):
        raise ValueError("legacy search request identity mismatch")
    pairs.sort(key=lambda pair: known[pair[1]["generation_config_sha256"]])
    failure_paths, failures = [p for p, _ in pairs], [f for _, f in pairs]
    failure = failures[-1]
    prefix_path = failure_paths[-1].parent / failure["prefix"]["path"]
    if hashlib.sha256(prefix_path.read_bytes()).hexdigest() != failure["prefix"]["sha256"]:
        raise ValueError("failure prefix identity mismatch")
    prefix = json.loads(gzip.decompress(prefix_path.read_bytes()))
    if len(prefix["moves"]) != failure["ply"] or any(
        f["sfen"] != failure["sfen"] for f in failures
    ):
        raise ValueError("failure history mismatch")
    rng = random.Random(config["seed"] + failure["game"])
    for _ in prefix["moves"]:
        rng.random()  # random.choices consumes exactly one draw even for forced prefix moves.
    ledger = Ledger(output)
    try:
        identity = task_identity(
            config, failure["game"], failure["ply"], failure["branch"], failure, prefix["moves"]
        )
        key, current = ledger.task(identity)
        attempts = [
            {
                "nodes": known[f["generation_config_sha256"]],
                "outcome": f["error_type"],
                "source": str(path),
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "legacy": True,
                "stdout_tail": f.get("stdout_tail", ""),
            }
            for path, f in pairs
        ]
        if current["attempts"] and current["attempts"] != attempts:
            raise ValueError("legacy attempt import conflicts with existing task history")
        checkpoint = {
            **prefix,
            "initial_sfen": initial_sfen,
            "rng": rng.getstate(),
            "ply": failure["ply"],
            "sfen": failure["sfen"],
            "status": "deferred",
            "legacy_prefix_sha256": failure["prefix"]["sha256"],
        }
        # The entire historical budget and checkpoint commit together, or neither does.
        with ledger.db:
            ledger.db.execute(
                "UPDATE tasks SET attempts=?,status='deferred' WHERE id=?", (_json(attempts), key)
            )
            ledger.db.execute(
                "INSERT OR IGNORE INTO checkpoints VALUES(?,?)",
                (failure["game"], _json(checkpoint)),
            )
        return {
            "game": failure["game"],
            "records": len(prefix["records"]),
            "task": key,
            "attempts": len(ledger.get(key)["attempts"]),
        }
    finally:
        ledger.close()


def export_queue(output: Path):
    from .evaluator_data import atomic, digest, encoded

    ledger = Ledger(output)
    try:
        tasks = [
            {
                **json.loads(identity),
                "task_id": key,
                "status": "inflight" if status == "running" else status,
                "attempts": json.loads(attempts),
            }
            for key, identity, status, attempts in ledger.db.execute(
                "SELECT id,identity,status,attempts FROM tasks WHERE status!='accepted' ORDER BY id"
            )
        ]
        report = {
            "schema": "open_shogiai_recovery_queue/v1",
            "generation_sha256": digest(output / "generation.json"),
            "tasks": tasks,
            "summary": ledger.summary(),
        }
        atomic(output / "recovery-queue.json", encoded(report))
        return report
    finally:
        ledger.close()


def migrate_generation(root: Path, parent_data: Path, output: Path, config: dict):
    """Reuse only hash-verified committed pairs; preserve parent adoption identity."""
    import os

    from .evaluator_data import atomic, digest, encoded

    parent_sha = digest(parent_data / "generation.json")
    if parent_sha != config["recovery_policy"]["parent_generation_sha256"]:
        raise ValueError("parent generation identity mismatch")
    references = []
    (output / "games").mkdir(parents=True, exist_ok=True)
    for receipt in sorted((parent_data / "games").glob("*.receipt.json")):
        saved = json.loads(receipt.read_text())
        shard = parent_data / "games" / f"{saved['game']:06d}.json.gz"
        if shard.is_symlink() or digest(shard) != saved["sha256"]:
            raise ValueError("parent committed shard corrupted")
        for source in (shard, receipt):
            destination = output / "games" / source.name
            if destination.exists():
                if digest(destination) != digest(source):
                    raise ValueError("migration destination mismatch")
            else:
                os.link(source, destination)
        raw = json.loads(gzip.decompress(shard.read_bytes()))
        for row in raw["records"]:
            deviation = row.get("deviation")
            for observation in (deviation, deviation.get("recovery") if deviation else None):
                if not isinstance(observation, dict) or not observation.get("failure_receipt"):
                    continue
                evidence = parent_data / observation["failure_receipt"]
                if digest(evidence) != observation["failure_sha256"]:
                    raise ValueError("legacy missing-label evidence corrupted")
                failure = json.loads(evidence.read_text())
                prefix = evidence.parent / failure["prefix"]["path"]
                if digest(prefix) != failure["prefix"]["sha256"]:
                    raise ValueError("legacy missing-label prefix corrupted")
                for source in (evidence, prefix):
                    destination = output / source.relative_to(parent_data)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists():
                        if digest(source) != digest(destination):
                            raise ValueError("inherited evidence destination mismatch")
                    else:
                        os.link(source, destination)
        references.append(
            {
                "game": saved["game"],
                "sha256": saved["sha256"],
                "rows": saved["rows"],
                "receipt_sha256": digest(receipt),
            }
        )
    report = {
        "parent_generation_sha256": parent_sha,
        "adoption": "legacy_exact_depth_conditions",
        "games": references,
    }
    atomic(output / "inherited.json", encoded(report))
    return report
