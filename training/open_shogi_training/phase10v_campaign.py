"""Phase 10V frozen campaign decisions and evidence gates; never starts long work.

The caller executes one returned action, records its immutable inputs/outputs, and
calls again. Candidate rejection advances the predeclared schedule. Integrity
failure always closes execution. Raw Arena verification uses the existing
rules-only replay infrastructure, independently pinned manifests and model bytes.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from open_shogi_training.phase10u_arena_evidence import (
    WARMUP,
    EvidenceError,
    artifact,
    digest,
    read_receipt,
    validate,
    validate_manifest,
)

CUTOFF = datetime.fromisoformat("2026-09-13T06:00:00+09:00")
DEADLINE = datetime.fromisoformat("2026-09-13T12:00:00+09:00")
STAGES = (100_000, 500_000, 1_000_000)
ORIGINS = frozenset(("pure_vs_handcrafted", "prior_model", "self_generated"))
MODES = ("equal_nodes", "equal_wall_clock")
FLOOR = 80 * 1024**3


class CampaignError(ValueError):
    """STOP_CLOSED: integrity failure, never scientific candidate rejection."""


class CandidateRejectedError(ValueError):
    """Preserve this attempt and advance; not an integrity STOP_CLOSED."""


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise CampaignError(reason)


def load_configs(root: Path) -> dict:
    values = {
        name: json.loads((root / "configs" / "phase10v" / f"{name}.json").read_text())
        for name in ("campaign", "data", "arena", "resources")
    }
    campaign, data, arena, resources = (values[k] for k in values)
    require(campaign["stages"] == list(STAGES), "training stage sequence changed")
    require(campaign["widths"] == [256, 512], "broad architecture sweep forbidden")
    require(campaign["new_work_cutoff"] == CUTOFF.isoformat(), "cutoff changed")
    require(campaign["deadline"] == DEADLINE.isoformat(), "deadline changed")
    require(campaign["final_holdout"] is False, "holdout execution unauthorized")
    require(campaign["candidate_failure"] == "advance", "candidate failure must advance")
    require(
        campaign["attempts_per_variant"] == ["base", "checkpoint", "calibration", "hard1"],
        "candidate schedule changed",
    )
    require(campaign["integrity_failure"] == "STOP_CLOSED", "integrity gate changed")
    require(data["trajectory_origins"] == sorted(ORIGINS), "trajectory coverage changed")
    require(data["missing_multipv"] == "mask_observed_only", "fabricated teacher scores")
    require(data["split_unit"] == "connected_provenance_component", "split unit changed")
    require(arena["modes"] == list(MODES), "both Arena modes mandatory")
    require(arena["selector"] == "min_conservative_score_both_modes", "offline selection")
    require(arena["denominator"] == "all_scheduled_games", "optimistic denominator")
    require(arena["bootstrap_unit"] == "unique_start", "game-level bootstrap forbidden")
    for key, expected in {
        "nodes": 2000,
        "movetime_ms": 100,
        "threads": 1,
        "hash_mib": 32,
        "depth_cap": 64,
        "max_plies": 256,
        "minimum_completion_fraction": 0.95,
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 20260913,
        "lower_percentile": 0.025,
        "development_games_per_mode": 400,
        "selfplay_gate_games_per_mode": 800,
        "objective_games_per_mode": 1600,
        "final_holdout": False,
        "same_model_hash_both_modes": True,
    }.items():
        require(arena[key] == expected, f"Arena control changed: {key}")
    require(
        arena["gates"]
        == {"limited_selfplay": 0.35, "full_selfplay": 0.48, "research_objective": 0.55},
        "strength gates changed",
    )
    require(resources["hard_free_space_floor_bytes"] == FLOOR, "disk floor changed")
    return values


def validate_partitions(partitions: dict[str, list[dict]]) -> dict[str, int]:
    """All roots, candidates and trajectory descendants share component membership."""
    require(
        set(partitions) == {"train", "validation", "calibration", "final_holdout"},
        "missing isolated partition",
    )
    components: dict[str, str] = {}
    positions: dict[str, str] = {}
    for split, rows in partitions.items():
        require(bool(rows), f"empty {split} partition")
        for row in rows:
            for field, owners in (("component_id", components), ("position_sha256", positions)):
                value = row[field]
                require(isinstance(value, str) and bool(value), f"missing {field}")
                require(value not in owners or owners[value] == split, "partition leakage")
                owners[value] = split
            require(bool(row["source_sha256"]), "missing provenance")
            if row.get("trajectory_origin") is not None:
                require(
                    split == "train" and row["trajectory_origin"] in ORIGINS,
                    "trajectory descendants must be train-only",
                )
                require(
                    row.get("teacher_relabelled") is True and bool(row.get("teacher_sha256")),
                    "trajectory teacher relabelling missing",
                )
    return {name: len(rows) for name, rows in partitions.items()}


def verify_dataset(root: Path, manifest: dict, trusted_sha256: str) -> dict:
    """Bind the campaign to the data module's full raw-evidence reconstruction.

    Both train and validation bytes are verified together for leakage. Final
    holdout positions are never loaded; isolation uses sealed hash metadata.
    The outer manifest contains data_receipt/train/validation artifact refs.
    """
    from open_shogi_training.phase10v_data import verify_training_inputs

    require(digest(manifest) == trusted_sha256, "reviewed dataset manifest changed")
    require(manifest["split"] in ("train", "validation"), "holdout dataset cannot be opened")
    for name in ("data_receipt", "train", "validation"):
        artifact(root, manifest[name])
    return verify_training_inputs(
        root / manifest["train"]["path"],
        root / manifest["validation"]["path"],
        root / manifest["data_receipt"]["path"],
        manifest["data_receipt"]["sha256"],
    )


@dataclass(frozen=True)
class ArenaScore:
    model_sha256: str
    mode: str
    score: float
    lower: float
    completed_fraction: float
    scheduled: int
    manifest_sha256: str
    schedule_sha256: str = ""


def prepare_arena_manifest(
    root: Path,
    *,
    candidate: dict,
    handcrafted: dict | None = None,
    opponent: dict | None = None,
    starts: dict,
    oracle: dict,
    mode: str,
    games: int = 400,
) -> dict:
    """Prepare development-only, reversed-color strength comparisons."""
    return _prepare_manifest(
        root,
        candidate=candidate,
        handcrafted=handcrafted,
        opponent=opponent,
        starts=starts,
        oracle=oracle,
        mode=mode,
        games=games,
    )


def prepare_trajectory_manifest(
    root: Path,
    *,
    candidate: dict,
    opponent: dict,
    starts: dict,
    oracle: dict,
    origin: str,
    mode: str = "equal_nodes",
    games: int = 200,
) -> dict:
    """Prepare train-only games for mandatory external teacher relabelling.

    These games never certify strength or authorize unassisted self-play.
    Self-generated trajectories may use the same pure model on both sides.
    No games are run and no execution authorization is created here.
    """
    require(origin in ORIGINS, "unknown trajectory origin")
    expected_formats = {
        "pure_vs_handcrafted": {"HANDCRAFTED"},
        "prior_model": {"OSAVAL02", "OSAVAL03"},
        "self_generated": {"OSAVAL03"},
    }
    require(
        opponent["adapter_format"] in expected_formats[origin],
        "trajectory origin/opponent mismatch",
    )
    return _prepare_manifest(
        root,
        candidate=candidate,
        opponent=opponent,
        starts=starts,
        oracle=oracle,
        mode=mode,
        games=games,
        trajectory_origin=origin,
    )


def _prepare_manifest(
    root: Path,
    *,
    candidate: dict,
    handcrafted: dict | None = None,
    opponent: dict | None = None,
    starts: dict,
    oracle: dict,
    mode: str,
    games: int,
    trajectory_origin: str | None = None,
) -> dict:
    """Build the existing runner's manifest from certified content-addressed identities.

    Callers pin the returned canonical digest before creating a later execution
    authorization. This function only validates existing files and makes a plan.
    The same start ordering and seeds apply to both modes and all candidates.
    """
    trajectory = trajectory_origin is not None
    require(mode in MODES, "unknown schedule mode")
    require(
        type(games) is int
        and (2 <= games <= 200 and games % 2 == 0 if trajectory else games in (400, 800, 1600)),
        "unauthorized Arena schedule",
    )
    require(candidate["adapter_format"] == "OSAVAL03", "Phase10V candidate format required")
    require((handcrafted is None) != (opponent is None), "exactly one opponent identity required")
    comparison = handcrafted if handcrafted is not None else opponent
    require(
        comparison["adapter_format"] in ("HANDCRAFTED", "OSAVAL02", "OSAVAL03"),
        "comparison opponent unsupported",
    )
    if comparison["adapter_format"] != "HANDCRAFTED" and trajectory_origin != "self_generated":
        require(
            candidate["model"]["sha256"] != comparison["model"]["sha256"],
            "prior comparison must bind a different model",
        )
    rows = json.loads(artifact(root, starts))
    rows = rows["starts"] if isinstance(rows, dict) else rows
    split = "train" if trajectory else "development"
    require(len(rows) >= (1 if trajectory else 200), "insufficient frozen starts")
    require(len({row["sfen"] for row in rows}) == len(rows), "duplicate starts")
    require(
        all(row.get("split") == split for row in rows), "wrong split or holdout starts forbidden"
    )
    clock = mode == "equal_wall_clock"
    controls = {
        "clock": "monotonic_ns",
        "threads": 1,
        "hash_mb": 32,
        "max_depth": 64,
        "nodes": None if clock else 2000,
        "movetime_ns": 100_000_000 if clock else None,
        "hard_timeout_ns": 1_000_000_000 if clock else 30_000_000_000,
        "max_plies": 256,
        "warmup": WARMUP,
        "search_options": {
            "configuration": "SearchConfig::default",
            "hash_mb": 32,
            "fresh_engine_per_move": True,
            "book": False,
        },
    }
    schedule = {}
    for index in range(games // 2):
        start = rows[index % min(len(rows), 400)]
        seed = 20260913 + index // min(len(rows), 400)
        pair_id = f"p{index:04}"
        for color in (0, 1):
            sides = (candidate, comparison) if color == 0 else (comparison, candidate)
            schedule[f"{pair_id}-{color}"] = {
                "initial_sfen": start["sfen"],
                "seed": seed,
                "pairing_id": pair_id,
                "controls": controls,
                "sides": dict(zip(("black", "white"), sides, strict=True)),
            }
    manifest = {
        "schema": "open_shogiai_phase10u_reviewed_arena_manifest/v1",
        "phase10v_split": split,
        "status": "READY_FOR_REVIEW_NOT_EXECUTED",
        "execution_authorized": False,
        "start_manifest": starts,
        "replay_oracle": oracle,
        "games": schedule,
        "planned_games": games,
        "process_parallelism": 1,
    }
    if trajectory:
        manifest.update(
            purpose="teacher_relabelled_trajectory",
            trajectory_origin=trajectory_origin,
            teacher_relabel_required=True,
            strength_evidence=False,
        )
    validate_manifest(root, manifest, digest(manifest))
    return manifest


def pair_statistics(groups: list[list[float | None]], *, replicates: int = 10_000) -> tuple:
    """Bootstrap unique starts, keeping all color/seed repetitions together.

    None means uncompleted/max-plies: zero numerator, scheduled denominator intact.
    """
    require(len(groups) >= 2 and replicates >= 100, "insufficient bootstrap evidence")
    require(all(len(group) >= 2 and len(group) % 2 == 0 for group in groups), "unpaired games")
    require(
        all(
            v is None or (type(v) in (float, int) and v in (0, 0.5, 1))
            for group in groups
            for v in group
        ),
        "invalid game score",
    )
    totals = [(sum(v or 0 for v in group), len(group)) for group in groups]
    scheduled = sum(n for _, n in totals)
    score = sum(s for s, _ in totals) / scheduled
    rng = random.Random(20260913)
    boot = []
    for _ in range(replicates):
        sample = rng.choices(totals, k=len(totals))
        boot.append(sum(s for s, _ in sample) / sum(n for _, n in sample))
    boot.sort()
    lower = boot[math.floor(0.025 * (replicates - 1))]
    completion = sum(v is not None for group in groups for v in group) / scheduled
    return score, lower, completion, scheduled


def verified_arena(
    root: Path,
    trusted: dict,
    trusted_sha256: str,
    receipt_paths: list[Path],
    model_sha256: str,
    mode: str,
) -> ArenaScore:
    """Verify actual raw receipt replay before computing any progression evidence."""
    require(mode in MODES, "unknown mode")
    require(trusted.get("phase10v_split") == "development", "final holdout is isolated")
    try:
        validate_manifest(root, trusted, trusted_sha256)
        receipts = {}
        for path in receipt_paths:
            receipt = read_receipt(path)
            require(receipt["game_id"] not in receipts, "duplicate receipt")
            validate(root, receipt, trusted, trusted_sha256)
            receipts[receipt["game_id"]] = receipt
        require(set(receipts) <= set(trusted["games"]), "unscheduled receipt")
        groups: dict[str, list[float | None]] = {}
        pairing: dict[str, list[tuple[str, str, int]]] = {}
        for game_id, game in trusted["games"].items():
            controls = game["controls"]
            require(
                (controls["nodes"] == 2000 and controls["movetime_ns"] is None)
                if mode == "equal_nodes"
                else (controls["nodes"] is None and controls["movetime_ns"] == 100_000_000),
                "Arena mode/control mismatch",
            )
            sides = game["sides"]
            candidates = [
                side
                for side, identity in sides.items()
                if identity["model"] is not None and identity["model"]["sha256"] == model_sha256
            ]
            require(len(candidates) == 1, "candidate model not uniquely bound")
            side = candidates[0]
            other = "white" if side == "black" else "black"
            require(sides[other]["adapter_format"] == "HANDCRAFTED", "wrong gate opponent")
            pairing.setdefault(game["pairing_id"], []).append(
                (game["initial_sfen"], side, game["seed"])
            )
            receipt = receipts.get(game_id)
            value = None
            if receipt is not None and not receipt["max_plies_reached"]:
                result = receipt["result"]
                require(result in ("black", "white", "draw"), "unknown result")
                value = 0.5 if result == "draw" else float(result == side)
            groups.setdefault(game["initial_sfen"], []).append(value)
        for pair in pairing.values():
            require(
                len(pair) == 2
                and pair[0][0] == pair[1][0]
                and pair[0][1] != pair[1][1]
                and pair[0][2] == pair[1][2],
                "schedule lacks reversed-color pairs",
            )
        schedule_sha256 = digest(
            sorted(
                (game["initial_sfen"], game["seed"], game["pairing_id"])
                for game in trusted["games"].values()
            )
        )
        return ArenaScore(
            model_sha256,
            mode,
            *pair_statistics(list(groups.values())),
            trusted_sha256,
            schedule_sha256,
        )
    except (EvidenceError, KeyError, TypeError, OSError) as error:
        raise CampaignError(f"STOP_CLOSED: {error}") from error


def strength(scores: list[ArenaScore], *, minimum_games: int = 400) -> float:
    require(len(scores) == 2 and {s.mode for s in scores} == set(MODES), "both modes required")
    require(len({s.model_sha256 for s in scores}) == 1, "different models between modes")
    require(len({s.schedule_sha256 for s in scores}) == 1, "different starts/seeds between modes")
    require(
        all(0 <= s.lower <= s.score <= 1 and 0 <= s.completed_fraction <= 1 for s in scores),
        "invalid statistics",
    )
    if not all(s.scheduled >= minimum_games and s.completed_fraction >= 0.95 for s in scores):
        raise CandidateRejectedError("insufficient completed Arena evidence; advance schedule")
    return min(s.score for s in scores)


def selfplay_permission(scores: list[ArenaScore], relabelled_origins: set[str]) -> str:
    value = strength(scores, minimum_games=800)
    require(
        relabelled_origins == ORIGINS, "complete teacher-relabelled trajectory coverage required"
    )
    if value >= 0.48 and min(s.lower for s in scores) >= 0.43:
        return "full"
    if value >= 0.35 and min(s.lower for s in scores) >= 0.30:
        return "limited"
    return "teacher_relabelled_only"


def select_candidate(candidates: dict[str, list[ArenaScore]]) -> str:
    """Strength first; deterministic hash tie-break; never inspect final holdout/offline loss."""
    require(bool(candidates), "no candidates")
    for key, scores in candidates.items():
        require(all(s.model_sha256 == key for s in scores), "selection model mismatch")
    return max(sorted(candidates), key=lambda key: strength(candidates[key]))


def next_action(state: dict, *, now: datetime, free_bytes: int) -> dict:
    """Read-only scheduling API. Completed steps are immutable receipt-backed IDs.

    Integrity preflight is caller-owned; failures must be supplied as integrity_errors.
    Scientific failures count as completed attempts and advance to the next ID.
    """
    require(now.tzinfo is not None, "timezone required")
    if state.get("integrity_errors"):
        return {"action": "STOP_CLOSED", "reasons": state["integrity_errors"]}
    if now >= DEADLINE:
        return {"action": "FINALIZE", "reason": "Sunday deadline"}
    if free_bytes < FLOOR:
        return {"action": "PRESERVE_AND_REVIEW_CLEANUP", "resume": "same_pending_step"}
    if now >= CUTOFF:
        return {"action": "FINALIZE", "reason": "no new long work after 06:00 JST"}
    completed = state.get("completed", {})
    require(isinstance(completed, dict), "receipt-backed completion map required")
    require(
        all(
            isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)
            for v in completed.values()
        ),
        "completion receipt digest missing",
    )
    steps = [("preflight", "VERIFY_IDENTITIES_SPLITS_ARTIFACTS")]
    for stage in STAGES:
        steps.append((f"{stage}/labels", "PREPARE_TEACHER_LABELS"))
        for variant in (256, 512):
            for attempt in ("base", "checkpoint", "calibration", "hard1"):
                prefix = f"{stage}/w{variant}/{attempt}"
                steps.extend(
                    [
                        (f"{prefix}/train", "TRAIN_CANDIDATE"),
                        (f"{prefix}/equal_nodes", "ARENA_EQUAL_NODES"),
                        (f"{prefix}/equal_wall_clock", "ARENA_EQUAL_CLOCK"),
                    ]
                )
        steps.append((f"{stage}/select", "SELECT_BY_PLAYING_STRENGTH"))
    for step_id, action in steps:
        if step_id not in completed:
            return {"action": action, "step_id": step_id}
    # Continue bounded train-only hard-example rounds until the cutoff. Larger
    # labels require a separately verified earlier Arena justification receipt.
    round_id = state.get("hard_round", 2)
    require(type(round_id) is int and round_id >= 2, "invalid hard round")
    for origin in sorted(ORIGINS):
        step_id = f"hard{round_id}/relabel/{origin}"
        if step_id not in completed:
            return {"action": "TEACHER_RELABEL_TRAIN_TRAJECTORIES", "step_id": step_id}
    for variant in (256, 512):
        for action in ("train", *MODES):
            step_id = f"hard{round_id}/w{variant}/{action}"
            if step_id not in completed:
                return {
                    "action": "TRAIN_CANDIDATE" if action == "train" else f"ARENA_{action.upper()}",
                    "step_id": step_id,
                }
    return {"action": "ADVANCE_HARD_ROUND", "hard_round": round_id + 1}


def larger_labeling_allowed(scores: list[ArenaScore]) -> bool:
    return strength(scores) >= 0.20 and min(s.lower for s in scores) >= 0.15


def verify_execution_state(root: Path, state: dict) -> None:
    """Resolve completion markers to immutable actual attempt receipt bytes.

    PASS must follow the action's dedicated validator. This journal verifier
    detects corruption/rebinding, not scientific validity of a claimed PASS.
    """
    for step_id, sha in state.get("completed", {}).items():
        ref = state["completion_receipts"][sha]
        require(ref["sha256"] == sha, "journal digest mismatch")
        receipt = json.loads(artifact(root, ref))
        require(receipt["step_id"] == step_id, "journal step mismatch")
        require(
            receipt["status"] in ("PASS", "CANDIDATE_REJECTED"),
            "failed integrity cannot mark completed",
        )
        if receipt["status"] == "CANDIDATE_REJECTED":
            require(
                step_id.endswith(("/train", "/equal_nodes", "/equal_wall_clock")),
                "prerequisite cannot be waived as candidate rejection",
            )
        require(bool(receipt["evidence"]), "journal evidence missing")
        for evidence_ref in receipt["evidence"]:
            artifact(root, evidence_ref)


def research_objective_met(scores: list[ArenaScore]) -> bool:
    return strength(scores, minimum_games=1600) >= 0.55 and min(s.lower for s in scores) > 0.50


def main() -> None:
    raise SystemExit("Closed campaign: see docs/status.md; use current development commands")


if __name__ == "__main__":
    main()
