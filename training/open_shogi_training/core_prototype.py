"""One bounded computation-policy fit; no leaf training or external teacher at play time."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import numpy as np

from .phase10v_data import position_hash

FEATURES = [
    "incumbent",
    "gap_cp_1000",
    "absolute_cp_1000",
    "delta_cp_1000",
    "node_share",
    "depth_8",
    "log_candidates",
    "in_check",
    "prior_incumbent",
    "root_volatility_cp_1000",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(sfen: str) -> str:
    """Move number is not a distinct position; preserve board, turn, and all hands."""
    fields = sfen.split()
    if len(fields) != 4 or fields[1] not in {"b", "w"}:
        raise ValueError("invalid canonical SFEN")
    return " ".join(fields[:3])


def partition(game: int) -> str:
    if not 0 <= game < 24:
        raise ValueError("fixed cohort contains only game IDs 0..23")
    return "train" if game < 16 else "validation" if game < 20 else "development_test"


def dataset(
    records: list[dict], excluded_hashes: set[str] | None = None
) -> tuple[dict[str, list], dict]:
    groups = {name: [] for name in ("train", "validation", "development_test")}
    seen: dict[str, str] = {}
    duplicates = 0
    skipped = 0
    excluded = 0
    positions = dict.fromkeys(groups, 0)
    source_games = {name: set() for name in groups}
    for record in records:
        sample, search = record["sample"], record["search"]
        split = partition(sample["game_id"])
        key = canonical(sample["sfen"])
        if position_hash(sample["sfen"]) in (excluded_hashes or set()):
            excluded += 1
            continue
        if key != canonical(search["sfen"]):
            raise ValueError("generator/search SFEN conversion mismatch")
        if key in seen:
            # Exclude the duplicate entirely even when it belongs to a different split.
            duplicates += 1
            continue
        seen[key] = split
        expected_side = "Black" if sample["sfen"].split()[1] == "b" else "White"
        if search["perspective"] != expected_side:
            raise ValueError("root score perspective mismatch")
        proof = search["proof"]
        for field in (
            "handcrafted_eval_calls",
            "residual_eval_calls",
            "composite_eval_calls",
            "book_hits",
            "teacher_calls",
            "fallback_count",
        ):
            if proof[field] != 0:
                raise ValueError(f"impure label search: {field}")
        if proof["learned_eval_calls"] <= 0 or not search["legal"]:
            raise ValueError("label search did not evaluate a legal nonterminal position")
        added = 0
        iterations = search["iterations"]
        for current, deeper in pairwise(iterations):
            if len({root["move"] for root in current["roots"]}) != len(current["roots"]):
                raise ValueError("duplicate aspiration candidates in completed search evidence")
            # Forced-terminal scores have separate semantics and are not logistic labels.
            if deeper["depth"] != current["depth"] + 1 or abs(deeper["score"]) >= 29000:
                continue
            incumbent = current["best_move"]
            winner = deeper["best_move"]
            if winner is None:
                continue
            for root in current["roots"]:
                features = root["features"]
                if len(features) != len(FEATURES) or not np.isfinite(features).all():
                    raise ValueError("invalid feature vector")
                if features[0] != float(root["move"] == incumbent):
                    raise ValueError("incumbent feature mismatch")
                expected_gap = np.clip(current["score"] - root["score"], -20000, 20000) / 1000
                if not np.isclose(features[1], expected_gap):
                    raise ValueError("score sign or centipawn scale mismatch")
                label = winner != incumbent if root["move"] == incumbent else root["move"] == winner
                groups[split].append((features, float(label), sample["game_id"], key))
                added += 1
        if added:
            positions[split] += 1
            source_games[split].add(sample["game_id"])
        else:
            skipped += 1
    if any(not rows for rows in groups.values()):
        raise ValueError("every predefined source-game partition needs completed-depth labels")
    return groups, {
        "acquired_positions": len(records),
        "deduplicated_positions": len(seen),
        "duplicate_positions_removed": duplicates,
        "positions_without_depth_pair": skipped,
        "frozen_metadata_overlap_excluded": excluded,
        "labeled_positions": positions,
        "source_games": {k: sorted(v) for k, v in source_games.items()},
        "examples": {k: len(v) for k, v in groups.items()},
        "cross_split_duplicate_retained": 0,
        "label_definition": (
            "candidate decision status changes at next completed depth; "
            "noisy search reference, not truth"
        ),
        "mate_labels": "excluded; abs(reference score) >= 29000",
    }


def probability(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x @ w + b, -30, 30)))


def metrics(y: np.ndarray, p: np.ndarray, incumbent: np.ndarray) -> dict:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return {
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier": float(np.mean((p - y) ** 2)),
        "incumbent_brier": float(np.mean((p[incumbent] - y[incumbent]) ** 2)),
        "incumbent_change_rate": float(np.mean(y[incumbent])),
        "predicted_incumbent_risk": float(np.mean(p[incumbent])),
        "positive_labels": int(y.sum()),
        "examples": len(y),
    }


def fit(groups: dict[str, list], leaf: str, steps: int = 300) -> tuple[dict, dict, dict]:
    if not 1 <= steps <= 300:
        raise ValueError("this prototype is capped at 300 full-batch updates")
    raw = {k: np.asarray([r[0] for r in v], dtype=np.float64) for k, v in groups.items()}
    labels = {k: np.asarray([r[1] for r in v], dtype=np.float64) for k, v in groups.items()}
    mean = raw["train"].mean(axis=0)
    scale = np.maximum(raw["train"].std(axis=0), 0.001)
    x = {k: np.clip((v - mean) / scale, -8, 8) for k, v in raw.items()}
    rate = float(labels["train"].mean())
    if not 0 < rate < 1:
        raise ValueError("no useful teacher signal: both labels are required")
    incumbent = {k: v[:, 0] == 1 for k, v in raw.items()}
    w = np.zeros(len(FEATURES))
    b = float(np.log(rate / (1 - rate)))
    best = None
    curve = []
    # One fixed optimizer/setting. Validation selects a checkpoint, never normalization.
    for step in range(1, steps + 1):
        error = probability(x["train"], w, b) - labels["train"]
        w -= 0.05 * (x["train"].T @ error / len(error) + 0.001 * w)
        b -= 0.05 * float(error.mean())
        if step % 10 == 0 or step == steps:
            score = metrics(
                labels["validation"], probability(x["validation"], w, b), incumbent["validation"]
            )
            curve.append({"step": step, "validation_log_loss": score["log_loss"]})
            if best is None or score["log_loss"] < best[0]:
                best = (score["log_loss"], step, w.copy(), b)
    assert best is not None
    _, best_step, best_w, best_b = best
    report = {}
    incumbent_prior = float(labels["train"][incumbent["train"]].mean())
    challenger_prior = float(labels["train"][~incumbent["train"]].mean())
    for split in groups:
        # Independent test is evaluated only here, after checkpoint selection is complete.
        report[split] = {
            "learned": metrics(
                labels[split], probability(x[split], best_w, best_b), incumbent[split]
            ),
            "constant_train_prior": metrics(
                labels[split], np.full(len(labels[split]), rate), incumbent[split]
            ),
            "role_conditioned_train_prior": metrics(
                labels[split],
                np.where(incumbent[split], incumbent_prior, challenger_prior),
                incumbent[split],
            ),
        }
    if (
        report["validation"]["learned"]["log_loss"]
        >= report["validation"]["constant_train_prior"]["log_loss"]
    ):
        raise ValueError(
            "learning did not improve validation loss; inspect format/signal "
            "instead of adding epochs"
        )
    artifact = {
        "schema": "open_shogiai_computation/v1",
        "leaf_model_sha256": leaf,
        "features": FEATURES,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weights": best_w.tolist(),
        "bias": best_b,
        "reference_risk": float(np.clip(labels["train"][incumbent["train"]].mean(), 0.01, 0.99)),
        "training_positions": len({r[3] for r in groups["train"]}),
        "updates": best_step,
    }
    return (
        artifact,
        {
            "updates_executed": steps,
            "selected_update": best_step,
            "example_exposures": len(groups["train"]) * steps,
            "position_exposures": artifact["training_positions"] * steps,
            "optimizer": "full-batch logistic gradient descent; lr=0.05, l2=0.001",
            "metrics": report,
            "curve": curve,
        },
        {"weights": w.tolist(), "bias": b, "updates": steps},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--leaf-sha256", required=True)
    parser.add_argument("--split-guard", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output already exists; do not silently replace an experiment")
    records = [json.loads(line) for line in args.records.read_text().splitlines()]
    if len(records) > 288 or any(r["search"]["leaf_sha256"] != args.leaf_sha256 for r in records):
        raise ValueError("cohort budget or leaf identity mismatch")
    guard = json.loads(args.split_guard.read_text())
    excluded_hashes = {
        row["position_sha256"]
        for split in ("train", "validation", "calibration", "final_holdout")
        for row in guard[split]
    }
    if not excluded_hashes or any(len(value) != 64 for value in excluded_hashes):
        raise ValueError("invalid frozen split metadata")
    groups, counts = dataset(records, excluded_hashes)
    artifact, training, resume = fit(groups, args.leaf_sha256)
    args.output.mkdir(parents=True)
    model_path = args.output / "controller.json"
    model_path.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")
    (args.output / "resume.json").write_text(json.dumps(resume, indent=2, allow_nan=False) + "\n")
    report = {
        "schema": "open_shogiai_core_training/v1",
        "records_sha256": digest(args.records),
        "model_sha256": digest(model_path),
        "leaf_sha256": args.leaf_sha256,
        "counts": counts,
        "split_guard_sha256": digest(args.split_guard),
        "training": training,
        "retained_checkpoints": 1,
        "resume_states": 1,
        "source": "own legal self-generated games; no third-party acquisition",
        "final_holdout_opened": False,
        "strength_claim": None,
    }
    (args.output / "training.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
