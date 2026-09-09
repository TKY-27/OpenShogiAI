from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "configs/runtime/pure_learned-v1.json"
REGISTRY = ROOT / "configs/models/registry.json"
ARENA = ROOT / "configs/phase10r/arena"
ARENA_GATES = ROOT / "configs/phase10r/arena-gates.yaml"
PROFILE_SCHEMA_SHA256 = "291cccea2056bee039fe4c84185304b3681ad6361df2ca1ba821a80d3645698c"
GIT_COMMIT = "2196edc2310f691b9a7ae9714a382d70e2a7e3f6"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_profile_contract_and_registry_are_hash_bound() -> None:
    profile = _load(PROFILE)
    registry = _load(REGISTRY)
    assert profile["schema"] == "open_shogiai_pure_learned_profile/v1"
    assert profile["profile"] == "pure_learned"
    assert profile["modelFormat"] == "OSAVAL02"
    for prohibited in (
        "handcraftedEvaluation",
        "residualEvaluation",
        "compositeEvaluation",
        "openingBook",
        "openingStyleRestriction",
        "teacherDuringPlay",
        "handcraftedFallback",
    ):
        assert profile[prohibited] is False
    assert registry["default_evaluator"] == "handcrafted-experimental"
    assert len(registry["models"]) == 1
    model = registry["models"][0]
    assert model["role"] == "frozen_comparison"
    assert model["default"] is False and model["distributed"] is False
    assert model["license_status"] == "pending-review"
    assert model["profile_sha256"] == _sha256(ROOT / model["profile"])


def _assert_valid_pure_learned_proof(proof: dict[str, object], model_sha256: str) -> None:
    assert proof["profile"] == "pure_learned"
    assert proof["profile_schema"] == "open_shogiai_pure_learned_profile/v1"
    assert proof["learned_eval_calls"] > 0
    for prohibited in (
        "handcrafted_eval_calls",
        "residual_eval_calls",
        "composite_eval_calls",
        "book_hits",
        "teacher_calls",
        "fallback_count",
    ):
        assert proof[prohibited] == 0, prohibited
    assert proof["model_sha256"] == model_sha256
    assert proof["evaluator_profile_schema_hash"] == PROFILE_SCHEMA_SHA256


def _deterministic_osaval02_artifact(tmp_path: Path) -> tuple[Path, str]:
    from open_shogi_training.phase10r_model import (
        VARIANT_PAIR,
        deterministic_test_tensors,
        serialize_osaval02,
    )

    artifact = serialize_osaval02(
        deterministic_test_tensors(VARIANT_PAIR),
        variant_id=VARIANT_PAIR,
        quantization="int8",
        training_run_reference="pure-learned-profile-test",
        git_commit=GIT_COMMIT,
    )
    model_path = tmp_path / "pure-learned-probe.osaval02"
    model_path.write_bytes(artifact)
    return model_path, hashlib.sha256(artifact).hexdigest()


def _run_arena(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["cargo", "run", "--quiet", "--locked", "-p", "open-shogi-cli", "--", "arena", *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_native_arena_pure_learned_reports_measured_runtime_proofs(tmp_path: Path) -> None:
    model_path, model_sha256 = _deterministic_osaval02_artifact(tmp_path)
    output_dir = tmp_path / "arena"
    completed = _run_arena(
        [
            "--games",
            "1",
            "--player-a",
            "pure_learned",
            "--a-model",
            os.fspath(model_path),
            "--a-model-sha256",
            model_sha256,
            "--a-opening-profile",
            "unrestricted",
            "--player-b",
            "handcrafted-experimental",
            "--nodes",
            "16",
            "--max-plies",
            "24",
            "--seed",
            "20260905",
            "--output-dir",
            os.fspath(output_dir),
        ]
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads((output_dir / "arena-report.json").read_text(encoding="utf-8"))
    assert report["run"]["playerA"]["evaluatorKind"] == "pure_learned"
    assert report["run"]["playerA"]["runtimeProfile"] == "pure_learned"
    assert report["run"]["playerB"]["runtimeProfile"] == "standard"
    aggregate = report["metrics"]["playerARuntimeProof"]
    _assert_valid_pure_learned_proof(aggregate, model_sha256)
    assert report["metrics"]["playerBRuntimeProof"] is None
    per_game = report["games"][0]["playerARuntimeProof"]
    _assert_valid_pure_learned_proof(per_game, model_sha256)
    assert per_game == aggregate


def test_native_arena_pure_learned_fails_closed_on_model_hash_mismatch(tmp_path: Path) -> None:
    model_path, model_sha256 = _deterministic_osaval02_artifact(tmp_path)
    wrong = "b" * 64
    assert wrong != model_sha256
    completed = _run_arena(
        [
            "--games",
            "1",
            "--player-a",
            "pure_learned",
            "--a-model",
            os.fspath(model_path),
            "--a-model-sha256",
            wrong,
            "--a-opening-profile",
            "unrestricted",
            "--player-b",
            "handcrafted-experimental",
            "--output-dir",
            os.fspath(tmp_path / "arena"),
        ]
    )
    assert completed.returncode != 0
    combined = completed.stdout + completed.stderr
    assert "SHA-256 mismatch" in combined


def _run_wasm_search(
    model_path: Path, expected_sha256: str, profile: str = "eco"
) -> tuple[int, str, str]:
    completed = subprocess.run(
        [
            "node",
            os.fspath(ROOT / "scripts/pure_learned_wasm_search.mjs"),
            os.fspath(model_path),
            expected_sha256,
            profile,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return completed.returncode, completed.stdout, completed.stderr


def test_wasm_pure_learned_search_matches_the_native_profile_identity(
    tmp_path: Path,
) -> None:
    model_path, model_sha256 = _deterministic_osaval02_artifact(tmp_path)
    code, out, err = _run_wasm_search(model_path, model_sha256)
    assert code == 0, err
    payload: dict[str, Any] = json.loads(out)
    response = payload["response"]
    assert response["evaluator"] == "pure_learned"
    _assert_valid_pure_learned_proof(response["runtimeProof"], model_sha256)
    assert response["runtimeProof"]["learned_eval_calls"] > 0
    # The book path is bypassed for the profile, so no book move key can appear.
    assert "openingBookMove" not in response
    # The measured stats summary (camelCase) must agree with the profile proof.
    assert response["stats"]["handcraftedEvalCalls"] == 0
    assert response["stats"]["learnedEvalCalls"] > 0


def test_wasm_pure_learned_without_a_verified_hash_fails_closed(tmp_path: Path) -> None:
    model_path, _ = _deterministic_osaval02_artifact(tmp_path)
    code, out, err = _run_wasm_search(model_path, "null")
    assert code != 0
    assert "pure_learned requires an explicitly verified model artifact SHA-256" in err + out


def test_usi_pure_learned_profile_selection_reports_a_valid_proof(tmp_path: Path) -> None:
    model_path, model_sha256 = _deterministic_osaval02_artifact(tmp_path)
    setup = "\n".join(
        [
            "usi",
            "setoption name RuntimeProfile value pure_learned",
            f"setoption name ModelPath value {model_path}",
            "setoption name ModelKind value osaval02-quantized",
            f"setoption name ExpectedModelSha256 value {model_sha256}",
            "isready",
            "usinewgame",
            "position startpos",
        ]
    )
    process = subprocess.Popen(
        ["cargo", "run", "--quiet", "--locked", "-p", "open-shogi-cli", "--", "usi"],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(setup + "\n")
        process.stdin.write("go movetime 500\n")
        process.stdin.flush()

        deadline = time.monotonic() + 120
        lines: list[str] = []
        while time.monotonic() < deadline:
            line = process.stdout.readline()
            if not line:
                raise AssertionError("USI session terminated before bestmove")
            lines.append(line.rstrip("\n"))
            if line.startswith("bestmove "):
                break
        process.stdin.write("quit\n")
        process.stdin.flush()
        process.wait(timeout=60)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=30)
    assert "readyok" in lines
    proof_line = next(
        (line for line in lines if line.startswith("info string runtime_proof ")), None
    )
    assert proof_line is not None
    fields = dict(
        token.split("=", 1)
        for token in proof_line.removeprefix("info string runtime_proof ").split(" ")
        if "=" in token
    )
    assert fields["profile"] == "pure_learned"
    assert fields["schema"] == "open_shogiai_pure_learned_profile/v1"
    assert fields["model_sha256"] == model_sha256
    assert fields["evaluator_profile_schema_hash"] == PROFILE_SCHEMA_SHA256
    assert int(fields["learned_eval_calls"]) > 0
    for prohibited in (
        "handcrafted_eval_calls",
        "residual_eval_calls",
        "composite_eval_calls",
        "book_hits",
        "teacher_calls",
        "fallback_count",
    ):
        assert int(fields[prohibited]) == 0, prohibited
    assert fields["valid"] == "true"
