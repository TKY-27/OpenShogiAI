"""Frozen Phase 10T controls and fail-closed progression; never starts training."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

BASE = "configs/phase10t/"
CONTROL_NAMES = ("model", "targets", "data", "arena", "training", "pure-build")
INTEGRITY = (
    "provenance",
    "leakage",
    "hashes",
    "legality",
    "parity",
    "pure_runtime",
    "model_loading",
    "target_semantics",
    "build_audit",
)
QUALITY = ("source_metrics", "calibration", "forgetting", "teacher_score", "ranking")
ANCESTOR = "b72817e5d7a37479a27cd4cb72f0dfce230479f7"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(root: Path, name: str) -> Path:
    relative = Path(name)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or (
            any("holdout" in part.lower() for part in relative.parts)
            and relative.as_posix() != "configs/phase10r/holdout-policy.yaml"
        )
    ):
        raise ValueError(f"unsafe evidence path: {name}")
    path = root / relative
    if any(
        (root / Path(*relative.parts[:i])).is_symlink() for i in range(1, len(relative.parts) + 1)
    ) or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"unsafe evidence link: {name}")
    return path


def tree_identity(root: Path, name: str) -> dict[str, Any]:
    directory = safe_path(root, name)
    if not directory.is_dir():
        raise ValueError(f"missing preserved tree: {name}")
    entries = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink in preserved tree: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            safe_path(root, relative)
            entries.append([relative, path.stat().st_size, sha256(path)])
    return {
        "path": name,
        "files": len(entries),
        "bytes": sum(row[1] for row in entries),
        "sha256": hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode()).hexdigest(),
    }


def load_json(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    result = json.loads(path.read_text(), object_pairs_hook=unique)
    if not isinstance(result, dict):
        raise ValueError("JSON object required")
    return result


def validate(root: Path, *, local: bool = False) -> dict[str, Any]:
    raise ValueError("Closed campaign; historical freeze is retired")


def arena_statistics(
    pairs: list[list[float | None]], *, replicates: int = 10000
) -> dict[str, float]:
    """Resample start pairs, retaining capped outcomes as missing, never as draws."""
    if not pairs or any(
        not isinstance(pair, list)
        or len(pair) not in (2, 4)
        or any(v is not None and (isinstance(v, bool) or v not in (0, 0.5, 1)) for v in pair)
        for pair in pairs
    ):
        raise ValueError("paired reversed-color outcomes required")
    sums = [sum(v for v in pair if v is not None) for pair in pairs]
    counts = [sum(v is not None for v in pair) for pair in pairs]
    total = sum(counts)
    if not total:
        raise ValueError("no finished games")
    rng = random.Random(20260907)
    samples = []
    for _ in range(replicates):
        indices = [rng.randrange(len(pairs)) for _ in pairs]
        denominator = sum(counts[i] for i in indices)
        samples.append(sum(sums[i] for i in indices) / denominator if denominator else 0.0)
    samples.sort()
    return {
        "score": sum(sums) / total,
        "conservative_score": sum(sums) / sum(map(len, pairs)),
        "completion": total / sum(map(len, pairs)),
        "lower": samples[int(0.025 * replicates)],
    }


def progression(receipt: dict[str, Any], arena: dict[str, Any]) -> dict[str, Any]:
    """Gate attested evidence. Runner must verify raw artifacts before making this receipt."""
    if not isinstance(receipt, dict) or not isinstance(receipt.get("integrity"), dict):
        return {"action": "STOP_CLOSED", "reason": "invalid integrity evidence"}
    for key in INTEGRITY:
        if receipt["integrity"].get(key) is not True:
            return {"action": "STOP_CLOSED", "reason": f"missing/failed integrity: {key}"}
    quality = receipt.get("quality")
    if (
        not isinstance(quality, dict)
        or set(quality) != set(QUALITY)
        or any(type(v) is not bool for v in quality.values())
    ):
        return {"action": "STOP_CLOSED", "reason": "missing or invalid quality evidence"}
    rounds = receipt.get("relabel_rounds", 0)
    if type(rounds) is not int or rounds < 0:
        return {"action": "STOP_CLOSED", "reason": "invalid relabel counter"}
    retry = "HARD_RELABEL_RETRAIN" if rounds < 2 else "REVIEW_REQUIRED"
    if receipt.get("stage") == "offline":
        model = receipt.get("model_sha256")
        if (
            not isinstance(model, str)
            or len(model) != 64
            or any(c not in "0123456789abcdef" for c in model)
            or "arenas" in receipt
        ):
            return {"action": "STOP_CLOSED", "reason": "invalid offline-only binding"}
        return {
            "action": "OFFLINE_PASSED" if all(quality.values()) else retry,
            "reason": "offline-only evidence; no Arena or selfplay authorization",
        }
    gates = {gate["id"]: gate for gate in arena["gates"]}
    if not isinstance(receipt.get("gate"), str) or receipt["gate"] not in gates:
        return {"action": "STOP_CLOSED", "reason": "unknown gate"}
    gate = gates[receipt["gate"]]
    evidence = receipt.get("arenas")
    if not isinstance(evidence, dict) or set(evidence) != {"equal_nodes", "equal_wall_clock"}:
        return {"action": "STOP_CLOSED", "reason": "both Arena modes required"}
    model_hash = receipt.get("model_sha256", "")
    if (
        not isinstance(model_hash, str)
        or len(model_hash) != 64
        or any(c not in "0123456789abcdef" for c in model_hash)
    ):
        return {"action": "STOP_CLOSED", "reason": "model identity missing"}
    # Validate every mode before considering quality. A weak first mode must not hide
    # corrupted evidence in the second mode. Repeated rounds share one start cluster.
    metrics = {}
    starts = None
    for mode, result in evidence.items():
        if (
            not isinstance(result, dict)
            or result.get("model_sha256") != model_hash
            or result.get("controls_verified") is not True
        ):
            return {"action": "STOP_CLOSED", "reason": "Arena binding missing"}
        pairs = result.get("pairs")
        ids = result.get("start_ids")
        cluster_games = 4 if receipt["gate"] == "final_objective" else 2
        if (
            not isinstance(pairs, list)
            or len(pairs) * cluster_games != gate["games_per_mode"]
            or any(not isinstance(pair, list) or len(pair) != cluster_games for pair in pairs)
        ):
            return {"action": "STOP_CLOSED", "reason": "wrong frozen game count"}
        if (
            not isinstance(ids, list)
            or len(ids) != len(pairs)
            or any(not isinstance(i, str) for i in ids)
            or len(set(ids)) != len(ids)
        ):
            return {"action": "STOP_CLOSED", "reason": "distinct start cluster identities required"}
        if starts is not None and starts != ids:
            return {"action": "STOP_CLOSED", "reason": "mismatched paired starts"}
        starts = ids
        try:
            stats = arena_statistics(pairs)
        except (ValueError, TypeError):
            return {"action": "STOP_CLOSED", "reason": "invalid outcomes"}
        if not all(math.isfinite(v) for v in stats.values()):
            return {"action": "STOP_CLOSED", "reason": "nonfinite metrics"}
        metrics[mode] = stats
    if not all(quality.values()):
        return {"action": retry, "reason": "quality failure; new train-only labels required"}
    for stats in metrics.values():
        lower_ok = (
            stats["lower"] > gate["lower"]
            if gate.get("lower_strict")
            else stats["lower"] >= gate["lower"]
        )
        if stats["completion"] < arena["minimum_completion_fraction"]:
            return {"action": retry, "reason": "excess capped games", "metrics": metrics}
        if stats["conservative_score"] < gate["score"] or not lower_ok:
            return {"action": retry, "reason": "strength gate missed", "metrics": metrics}
    return {
        "action": "REVIEW_BEFORE_FINAL_PROMOTION"
        if receipt["gate"] == "final_objective"
        else "GATE_PASSED",
        "gate": receipt["gate"],
        "metrics": metrics,
        "final_holdout_allowed": False,
        "promotion_allowed": False,
    }


def main() -> None:
    raise SystemExit("Closed campaign; use current development commands")


if __name__ == "__main__":
    main()
