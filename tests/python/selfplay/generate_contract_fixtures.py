"""Deterministically regenerate the shared Phase 6 contract fixtures.

The generator deliberately delegates all domain construction and validation to the
same production builders used by the tests.  Its only responsibilities are fixed
fixture inputs, canonical serialization, and the bounded base64/gzip envelope.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from conftest import tiny_model_bytes, write_complete_arena_evidence, write_ref
from open_shogi_training.selfplay.arena import (
    analyze_arena_results,
    decide_promotion,
    validate_promotion_decision_binding,
)
from open_shogi_training.selfplay.common import (
    ArtifactRef,
    artifact_ref,
    canonical_json_bytes,
)
from open_shogi_training.selfplay.config import load_generation_policy
from open_shogi_training.selfplay.registry import (
    build_initial_registry,
    record_generation_promotion,
    register_challenger_generation,
    validate_model_registry,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_ROOT = Path("tests/python/selfplay/fixtures/contracts")
REGISTRY_FIXTURE = Path("tests/python/selfplay/fixtures/registry-chain/registry-chain.json.gz.b64")
POLICY_PATH = Path("configs/generation/phase6_promotion.toml")
_OUTCOMES = {
    "promoted": "challenger",
    "rejected": "champion",
    "inconclusive": "balanced",
}
_CANONICAL_NUMBER_INPUTS = (
    ("small-negative-exponent", "1e-7"),
    ("threshold-exponent", "1e-6"),
    ("negative-zero", "-0.0"),
    ("positive-float", "1.0"),
    ("large-exponent", "1e+21"),
)


def _fixture_ref(path: Path, contents: bytes) -> ArtifactRef:
    return ArtifactRef(
        path=path.as_posix(),
        sha256=hashlib.sha256(contents).hexdigest(),
        size=len(contents),
    )


def _build_generation(
    root: Path,
    *,
    arena_outcome: str,
) -> tuple[dict[str, Any], dict[str, Any], ArtifactRef, dict[str, Any], ArtifactRef]:
    champion_bytes = tiny_model_bytes()
    challenger_bytes = tiny_model_bytes()
    champion = write_ref(root, "weights/champion.osaval", champion_bytes)
    challenger = write_ref(root, "weights/challenger.osaval", challenger_bytes)
    registry = build_initial_registry(
        generation_id="generation-0",
        champion_model_id="champion-v0",
        champion_artifact=champion,
        evaluator_kind="neural",
        architecture_version="1",
        quantization="float32",
        training_run=None,
        registered_at="2026-08-08T00:00:00Z",
    )
    registry = register_challenger_generation(
        registry,
        repository_root=root,
        generation_id="generation-0001",
        parent_generation_id="generation-0",
        challenger_model_id="challenger-v1",
        challenger_artifact=challenger,
        architecture_version="1",
        quantization="float32",
        selfplay_manifest=write_ref(root, "artifacts/selfplay.json", b"selfplay\n"),
        teacher_labeling_manifest=write_ref(root, "artifacts/labels.json", b"labels\n"),
        training_run_manifest=write_ref(root, "artifacts/training.json", b"training\n"),
        registered_at="2026-08-08T01:00:00Z",
    )
    results, results_ref = write_complete_arena_evidence(
        root,
        registry=registry,
        champion_ref=champion,
        champion_bytes=champion_bytes,
        challenger_ref=challenger,
        challenger_bytes=challenger_bytes,
        outcome=arena_outcome,
    )
    analysis = analyze_arena_results(results, results_ref=results_ref)
    analysis_ref = write_ref(
        root,
        "artifacts/analysis.json",
        canonical_json_bytes(analysis),
    )
    policy = load_generation_policy(PROJECT_ROOT / POLICY_PATH)
    policy_ref = write_ref(
        root,
        POLICY_PATH.as_posix(),
        (PROJECT_ROOT / POLICY_PATH).read_bytes(),
    )
    decision = decide_promotion(
        analysis,
        analysis_ref=analysis_ref,
        policy=policy,
        policy_ref=policy_ref,
        decided_at="2026-08-08T01:01:00Z",
    )
    decision_ref = write_ref(
        root,
        "artifacts/promotion-valid.json",
        canonical_json_bytes(decision),
    )
    final_registry = record_generation_promotion(
        registry,
        generation_id="generation-0001",
        arena_manifest=results_ref,
        promotion_decision=decision_ref,
        decision=str(decision["decision"]),
        repository_root=root,
    )
    final_registry_ref = write_ref(
        root,
        "weights/model-registry-final.json",
        canonical_json_bytes(final_registry),
    )
    validate_model_registry(final_registry, repository_root=root)
    return results, final_registry, final_registry_ref, decision, policy_ref


def _contract_outputs() -> dict[Path, bytes]:
    outputs: dict[Path, bytes] = {}
    policy = load_generation_policy(PROJECT_ROOT / POLICY_PATH)
    policy_ref = artifact_ref(PROJECT_ROOT, POLICY_PATH.as_posix())
    outcome_refs: dict[str, dict[str, object]] = {}
    for expected, arena_outcome in _OUTCOMES.items():
        with tempfile.TemporaryDirectory(prefix="open-shogi-contract-fixture.") as raw:
            results, _, _, _, _ = _build_generation(
                Path(raw).resolve(strict=True),
                arena_outcome=arena_outcome,
            )
        results_path = CONTRACT_ROOT / f"{expected}-results.json"
        results_bytes = canonical_json_bytes(results)
        results_ref = _fixture_ref(results_path, results_bytes)
        analysis = analyze_arena_results(results, results_ref=results_ref)
        analysis_path = CONTRACT_ROOT / f"{expected}-analysis.json"
        analysis_bytes = canonical_json_bytes(analysis)
        analysis_ref = _fixture_ref(analysis_path, analysis_bytes)
        decision = decide_promotion(
            analysis,
            analysis_ref=analysis_ref,
            policy=policy,
            policy_ref=policy_ref,
            decided_at="2026-08-08T01:01:00Z",
        )
        if decision.get("decision") != expected:
            raise RuntimeError(f"fixture outcome drifted: expected {expected!r}")
        validate_promotion_decision_binding(
            decision,
            analysis=analysis,
            analysis_ref=analysis_ref,
            policy=policy,
            policy_ref=policy_ref,
        )
        decision_path = CONTRACT_ROOT / f"{expected}-decision.json"
        decision_bytes = canonical_json_bytes(decision)
        decision_ref = _fixture_ref(decision_path, decision_bytes)
        outputs[results_path] = results_bytes
        outputs[analysis_path] = analysis_bytes
        outputs[decision_path] = decision_bytes
        outcome_refs[expected] = {
            "results": results_ref.as_dict(),
            "analysis": analysis_ref.as_dict(),
            "decision": decision_ref.as_dict(),
        }

    canonical_numbers = []
    for name, input_json in _CANONICAL_NUMBER_INPUTS:
        encoded = canonical_json_bytes(json.loads(input_json), newline=False)
        canonical_numbers.append(
            {
                "name": name,
                "inputJson": input_json,
                "expectedCanonical": encoded.decode("ascii"),
                "expectedSha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
    index = {
        "schema": "phase6_cross_runtime_contract_fixtures/v1",
        "policy": policy_ref.as_dict(),
        "policySha256": policy.sha256,
        "outcomes": outcome_refs,
        "canonicalNumbers": canonical_numbers,
    }
    outputs[CONTRACT_ROOT / "fixture-index.json"] = canonical_json_bytes(index)
    return outputs


def _registry_envelope() -> tuple[bytes, dict[str, int | str]]:
    with tempfile.TemporaryDirectory(prefix="open-shogi-registry-fixture.") as raw:
        root = Path(raw).resolve(strict=True)
        _, _, registry_ref, decision, _ = _build_generation(
            root,
            arena_outcome="balanced",
        )
        files: list[dict[str, object]] = []
        blobs: dict[str, str] = {}
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(root).as_posix()
            contents = path.read_bytes()
            sha256 = hashlib.sha256(contents).hexdigest()
            files.append({"path": relative, "sha256": sha256, "size": len(contents)})
            blobs.setdefault(sha256, base64.b64encode(contents).decode("ascii"))
        envelope = {
            "schema": "phase6_registry_acceptance_fixture/v1",
            "expectedDecision": decision["decision"],
            "registry": registry_ref.as_dict(),
            "files": files,
            "blobs": blobs,
        }
        envelope_bytes = canonical_json_bytes(envelope)
    compressed = gzip.compress(envelope_bytes, compresslevel=9, mtime=0)
    encoded = base64.b64encode(compressed).decode("ascii")
    lines = (encoded[index : index + 76] for index in range(0, len(encoded), 76))
    fixture_bytes = ("\n".join(lines) + "\n").encode("ascii")
    metadata: dict[str, int | str] = {
        "fixtureSize": len(fixture_bytes),
        "fixtureSha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "compressedSize": len(compressed),
        "compressedSha256": hashlib.sha256(compressed).hexdigest(),
        "envelopeSize": len(envelope_bytes),
        "envelopeSha256": hashlib.sha256(envelope_bytes).hexdigest(),
        "files": len(files),
        "blobs": len(blobs),
    }
    return fixture_bytes, metadata


def build_outputs() -> tuple[dict[Path, bytes], dict[str, int | str]]:
    outputs = _contract_outputs()
    registry_bytes, metadata = _registry_envelope()
    outputs[REGISTRY_FIXTURE] = registry_bytes
    return outputs, metadata


def _write_outputs(outputs: Mapping[Path, bytes]) -> None:
    for relative, contents in outputs.items():
        path = PROJECT_ROOT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)


def _check_outputs(outputs: Mapping[Path, bytes]) -> None:
    drifted = [
        relative.as_posix()
        for relative, contents in outputs.items()
        if not (PROJECT_ROOT / relative).is_file()
        or (PROJECT_ROOT / relative).read_bytes() != contents
    ]
    if drifted:
        raise SystemExit("fixture drift: " + ", ".join(drifted))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    outputs, metadata = build_outputs()
    if args.write:
        _write_outputs(outputs)
    else:
        _check_outputs(outputs)
    for key in sorted(metadata):
        print(f"{key}={metadata[key]}")
    for path in sorted(outputs):
        contents = outputs[path]
        sha256 = hashlib.sha256(contents).hexdigest()
        print(f"{path.as_posix()} size={len(contents)} sha256={sha256}")


if __name__ == "__main__":
    main()
