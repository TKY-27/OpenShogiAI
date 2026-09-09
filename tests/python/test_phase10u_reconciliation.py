"""Regression contracts for the reconciled, still-blocked incumbent selection."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_incumbent_uses_all_candidates_and_scheduled_denominators():
    manifest = json.loads((ROOT / "configs/phase10u/incumbent.json").read_text())
    scores = {row["variant"]: row for row in manifest["historical_scores"]}
    ranks = {}
    for prefix in ("candidate", "prior", "hard-candidate"):
        clock = scores[f"{prefix}-vs-handcrafted-equal-clock"]
        nodes = scores[f"{prefix}-vs-handcrafted-equal-nodes"]
        for row in (clock, nodes):
            total = row["wins"] + row["draws"] + row["losses"] + row["max_plies_excluded"]
            assert total == row["scheduled_games"] == 200
            assert (row["wins"] + row["draws"] / 2) / total == row[
                "conservative_score_all_scheduled"
            ]
        ranks[prefix] = (
            clock["conservative_score_all_scheduled"],
            clock["paired_bootstrap"]["lower_2_5_percentile"],
            nodes["conservative_score_all_scheduled"],
        )
    assert ranks["prior"][:2] == ranks["candidate"][:2]
    assert max(ranks, key=ranks.get) == "prior"
    assert tuple(manifest["measured_pure_incumbent"]["selection_tuple"]) == ranks["prior"]
    assert manifest["status"] == "STOP_CLOSED"
    assert manifest["fully_verified_sunday_candidate"] is None


def test_rerun_replaces_ranking_evidence_without_authorizing_execution():
    plan = json.loads((ROOT / "configs/phase10u/arena-rerun-plan.json").read_text())
    incumbent = json.loads((ROOT / "configs/phase10u/incumbent.json").read_text())
    assert plan["status"] in {"STOP_CLOSED_NOT_EXECUTED", "READY_FOR_REVIEW_NOT_EXECUTED"}
    if plan["status"] == "READY_FOR_REVIEW_NOT_EXECUTED":
        assert plan["implementation_review"]["adapter_audit"]["sha256"]
        assert plan["implementation_review"]["reviewed_manifest"]["sha256"]
        assert plan["implementation_review"]["validation"]["sha256"]
    assert plan["execution_authorized_in_this_task"] is False
    assert len({start["position_id"] for start in plan["starts"]}) == 100
    assert plan["games_per_variant"] == 2 * len(plan["starts"])
    assert (
        plan["total_new_games"]
        == 1200
        == (len(plan["entrants"]) * len(plan["modes"]) * plan["games_per_variant"])
    )
    assert plan["entrants"][0]["model"]["sha256"] == incumbent["measured_pure_incumbent"]["sha256"]
    assert plan["statistics"]["conservative_denominator"] == plan["games_per_variant"]
    assert plan["process_parallelism"] == 1
