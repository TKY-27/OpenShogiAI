"""Frozen Sunday decisions over verified runner receipts; never executes a campaign.

Integrity flags are attestations: callers must verify raw artifacts before issuing
receipts. This module validates bindings/decisions, not the truth of attestations.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from . import phase10t

BASE = "configs/phase10u/"
MODES = ("equal_nodes", "equal_wall_clock")
OPPONENTS = ("learned_1m", "handcrafted_experimental")
IMPLEMENTATION = (
    "training_target_masks",
    "game_lineage_split",
    "bounded_arena_runner",
    "teacher_relabelled_trajectory_ingestion",
)


def validate(root: Path, *, local: bool = False) -> dict[str, Any]:
    raise ValueError("Closed campaign; historical freeze is retired")


def campaign() -> dict[str, Any]:
    return {
        "schema": "open_shogiai_phase10u_campaign/v1",
        "status": "closed_historical_regression_fixture",
        "implementation_prerequisites": list(IMPLEMENTATION),
        "runner_status": "missing_paths_require_bounded_wiring_and_verification_before_execution",
        "deadline": "2026-09-13T12:00:00+09:00",
        "training_cutoff": "2026-09-13T06:00:00+09:00",
        "final_reserve_hours": 6,
        "checkpoints": [
            "100k_a1_export",
            "200_game_diagnostic",
            "1m_a1_export",
            "400_game_diagnostic",
            "strongest_sunday_candidate",
        ],
        "diagnostic_games_per_mode_per_opponent": [200, 400],
        "opponents": list(OPPONENTS),
        "modes": list(MODES),
        "arena_controls": "configs/phase10t/arena.json",
        "minimum_completion": 0.95,
        "improvement": (
            "learned_1m_clock_lower_and_conservative_gt_0.50_nodes_conservative_ge_0.50_"
            "and_no_handcrafted_regression_both_modes"
        ),
        "baseline": "fixed_learned_1m_vs_handcrafted_on_identical_starts",
        "caps": "missing_not_draw_conservative_score_divides_by_all_scheduled_games",
        "integrity": list(phase10t.INTEGRITY),
        "hard_relabel": {
            "max_rounds": 2,
            "max_positions_per_round": 25000,
            "nodes": 400000,
            "train_only": True,
        },
        "pre48_trajectories": (
            "pure_models_only_selected_train_positions_teacher_relabelled_before_training"
        ),
        "trajectory_limits": {
            "max_games_per_round": 200,
            "max_plies_per_game": 256,
            "max_selected_positions_per_round": 25000,
            "max_rounds_per_rung": 2,
        },
        "unassisted_selfplay": "phase10t_selfplay_entry_48pct_800_per_mode_lower43",
        "research_objective": "unchanged_phase10t_final55_1600_per_mode_lower_strictly_gt50",
        "sunday_candidate_requires_final55": False,
        "candidate_selection": (
            "all_attempts_reported_select_by_equal_clock_handcrafted_"
            "conservative_then_lower_then_equal_node_conservative"
        ),
        "final_holdout": False,
        "promotion": False,
    }


def _hash(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _stop(reason: str) -> dict[str, Any]:
    return {"action": "STOP_CLOSED", "reason": reason}


def decision(receipt: dict[str, Any], *, now: datetime) -> dict[str, Any]:
    """Bounded recommendation only; cannot authorize training or establish strength."""
    if now.tzinfo is None:
        return _stop("timezone-aware clock required")
    integrity = receipt.get("integrity")
    if not isinstance(integrity, dict) or any(
        integrity.get(k) is not True for k in phase10t.INTEGRITY
    ):
        return _stop("missing or failed integrity")
    quality = receipt.get("quality")
    if (
        not isinstance(quality, dict)
        or set(quality) != set(phase10t.QUALITY)
        or any(type(v) is not bool for v in quality.values())
    ):
        return _stop("missing or invalid quality")
    if not _hash(receipt.get("model_sha256")):
        return _stop("missing model binding")
    rounds = receipt.get("relabel_rounds")
    if type(rounds) is not int or not 0 <= rounds <= 2:
        return _stop("invalid repair counter")
    retry = "HARD_RELABEL_RETRAIN" if rounds < 2 else "REVIEW_REQUIRED"
    if now >= datetime.fromisoformat(campaign()["deadline"]):
        return {"action": "DEADLINE_CLOSED", "research_objective_completed": False}
    if now >= datetime.fromisoformat(campaign()["training_cutoff"]):
        return {
            "action": "EXPORT_INTEGRATE_HUMAN_EVALUATE",
            "training_allowed": False,
            "research_objective_completed": False,
        }
    stage = receipt.get("stage")
    implementation = receipt.get("implementation_verified")
    required = list(IMPLEMENTATION[:2])
    if stage in ("200_game_diagnostic", "400_game_diagnostic"):
        required.append("bounded_arena_runner")
    if stage == "teacher_relabelled_trajectories":
        required.append("teacher_relabelled_trajectory_ingestion")
    if not isinstance(implementation, dict) or any(
        implementation.get(k) is not True for k in required
    ):
        return _stop("missing runner implementation verification")
    if stage == "offline":
        if "arenas" in receipt:
            return _stop("offline receipt cannot contain Arena evidence")
        return {"action": "BOUNDED_DIAGNOSTIC" if all(quality.values()) else retry}
    if stage == "teacher_relabelled_trajectories":
        keys = (
            "pure_models_only",
            "train_only",
            "split_safe",
            "teacher_relabelled",
            "teacher_identity_verified",
            "unassisted_outcomes_excluded",
            "selected_positions_only",
        )
        if any(receipt.get(k) is not True for k in keys):
            return _stop("trajectory relabel boundary incomplete")
        limits = campaign()["trajectory_limits"]
        for field, limit in (
            ("trajectory_games", limits["max_games_per_round"]),
            ("trajectory_max_plies", limits["max_plies_per_game"]),
            ("selected_positions", limits["max_selected_positions_per_round"]),
        ):
            value = receipt.get(field)
            if type(value) is not int or not 0 < value <= limit:
                return _stop(f"invalid or exceeded trajectory bound: {field}")
        if rounds >= limits["max_rounds_per_rung"]:
            return _stop("trajectory repair rounds exhausted")
        return {
            "action": "RELABELLED_TRAJECTORIES_ELIGIBLE" if all(quality.values()) else retry,
            "unassisted_selfplay_allowed": False,
        }
    if stage in ("selfplay_entry", "final_objective"):
        return _stop("use unchanged phase10t progression with its frozen Arena control")
    if stage not in ("200_game_diagnostic", "400_game_diagnostic"):
        return _stop("unknown checkpoint")
    games = 200 if stage == "200_game_diagnostic" else 400
    opponents = receipt.get("opponent_sha256")
    evidence = receipt.get("arenas")
    baseline = receipt.get("baseline_arenas")
    if (
        not isinstance(opponents, dict)
        or set(opponents) != set(OPPONENTS)
        or not all(_hash(v) for v in opponents.values())
        or len(set(opponents.values()) | {receipt["model_sha256"]}) != 3
        or not isinstance(evidence, dict)
        or set(evidence) != set(OPPONENTS)
        or not isinstance(baseline, dict)
        or set(baseline) != set(MODES)
    ):
        return _stop("missing distinct fixed opponent/baseline bindings")
    metrics = {}
    starts = None
    groups = [
        (opponent, evidence[opponent], receipt["model_sha256"], opponents[opponent])
        for opponent in OPPONENTS
    ]
    groups.append(
        ("baseline", baseline, opponents["learned_1m"], opponents["handcrafted_experimental"])
    )
    for name, group, model, opponent in groups:
        if not isinstance(group, dict) or set(group) != set(MODES):
            return _stop("both diagnostic modes required")
        metrics[name] = {}
        for mode, result in group.items():
            if (
                not isinstance(result, dict)
                or result.get("model_sha256") != model
                or result.get("opponent_sha256") != opponent
                or result.get("controls_verified") is not True
            ):
                return _stop("diagnostic identity/control mismatch")
            ids, pairs = result.get("start_ids"), result.get("pairs")
            if (
                not isinstance(ids, list)
                or len(ids) != games // 2
                or any(not isinstance(i, str) or not i for i in ids)
                or len(set(ids)) != len(ids)
                or (starts is not None and ids != starts)
                or not isinstance(pairs, list)
                or len(pairs) != len(ids)
                or any(not isinstance(p, list) or len(p) != 2 for p in pairs)
            ):
                return _stop("wrong paired schedule")
            starts = ids
            try:
                metrics[name][mode] = phase10t.arena_statistics(pairs)
            except (ValueError, TypeError):
                return _stop("invalid game outcomes")
    if any(m["completion"] < 0.95 for group in metrics.values() for m in group.values()):
        return {
            "action": "REPEAT_BOUNDED_DIAGNOSTIC",
            "reason": "completion below 95%",
            "metrics": metrics,
        }
    learned = metrics["learned_1m"]
    improved = (
        learned["equal_wall_clock"]["lower"] > 0.50
        and learned["equal_wall_clock"]["conservative_score"] > 0.50
        and learned["equal_nodes"]["conservative_score"] >= 0.50
        and all(
            metrics["handcrafted_experimental"][m]["conservative_score"]
            >= metrics["baseline"][m]["conservative_score"]
            for m in MODES
        )
    )
    hard = receipt.get("hard_example_repair")
    justified = (
        isinstance(hard, dict)
        and rounds < 2
        and hard.get("train_only") is True
        and hard.get("split_safe") is True
        and hard.get("teacher_identity_verified") is True
        and hard.get("nodes") == 400000
        and type(hard.get("positions")) is int
        and 0 < hard["positions"] <= 25000
        and _hash(hard.get("selection_manifest_sha256"))
        and isinstance(hard.get("failure_analysis"), str)
        and bool(hard["failure_analysis"].strip())
    )
    if hard is not None and not justified:
        return _stop("invalid hard-example repair justification")
    if not all(quality.values()):
        action = "EXPAND_1M_WITH_HARD_RELABEL_RETRAIN" if justified and games == 200 else retry
    elif games == 200:
        action = (
            "EXPAND_1M"
            if improved
            else ("EXPAND_1M_WITH_HARD_RELABEL_RETRAIN" if justified else retry)
        )
    else:
        action = "SUNDAY_CANDIDATE_ELIGIBLE" if improved else retry
    return {
        "action": action,
        "strength_improved": improved,
        "metrics": metrics,
        "research_objective_completed": False,
        "unassisted_selfplay_allowed": False,
        "promotion_allowed": False,
        "final_holdout_allowed": False,
    }


def main() -> None:
    raise SystemExit("Closed campaign; use current development commands")


if __name__ == "__main__":
    main()
