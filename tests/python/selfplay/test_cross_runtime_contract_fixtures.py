from __future__ import annotations

import hashlib
import json
from pathlib import Path

from open_shogi_training.selfplay.arena import (
    analyze_arena_results,
    validate_promotion_decision_binding,
)
from open_shogi_training.selfplay.common import (
    ArtifactRef,
    canonical_json_bytes,
    load_bytes_artifact,
    load_json,
    load_json_artifact,
)
from open_shogi_training.selfplay.config import parse_generation_policy_bytes

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INDEX_PATH = Path("tests/python/selfplay/fixtures/contracts/fixture-index.json")


def test_cross_runtime_promotion_fixtures_are_exact_python_contract_outputs() -> None:
    index = load_json(PROJECT_ROOT / INDEX_PATH)
    assert index["schema"] == "phase6_cross_runtime_contract_fixtures/v1"
    policy_ref = ArtifactRef.from_dict(index["policy"], "fixture policy")
    policy = parse_generation_policy_bytes(
        load_bytes_artifact(PROJECT_ROOT, policy_ref, maximum_bytes=64 * 1024),
        policy_ref.path,
    )
    assert policy.sha256 == index["policySha256"]

    for expected_outcome, references in index["outcomes"].items():
        results_ref = ArtifactRef.from_dict(references["results"], "fixture results")
        analysis_ref = ArtifactRef.from_dict(references["analysis"], "fixture analysis")
        decision_ref = ArtifactRef.from_dict(references["decision"], "fixture decision")
        results = load_json_artifact(PROJECT_ROOT, results_ref)
        analysis = load_json_artifact(PROJECT_ROOT, analysis_ref)
        decision = load_json_artifact(PROJECT_ROOT, decision_ref)
        assert analysis == analyze_arena_results(results, results_ref=results_ref)
        validated = validate_promotion_decision_binding(
            decision,
            analysis=analysis,
            analysis_ref=analysis_ref,
            policy=policy,
            policy_ref=policy_ref,
        )
        assert validated["decision"] == expected_outcome


def test_cross_runtime_canonical_number_spellings_and_hashes_match_python() -> None:
    index = load_json(PROJECT_ROOT / INDEX_PATH)
    for case in index["canonicalNumbers"]:
        value = json.loads(case["inputJson"])
        encoded = canonical_json_bytes(value, newline=False)
        assert encoded.decode("ascii") == case["expectedCanonical"]
        assert hashlib.sha256(encoded).hexdigest() == case["expectedSha256"]
