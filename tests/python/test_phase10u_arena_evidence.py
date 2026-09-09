"""Bounded two-ply fixtures, never an Arena or model search."""

import copy
import functools
import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from open_shogi_training.phase10t_model import Phase10TModel
from open_shogi_training.phase10u_arena_evidence import (
    PROHIBITED,
    SCHEMA,
    EvidenceError,
    MonotonicSearch,
    canonical,
    digest,
    read_receipt,
    seal,
    validate,
    write_immutable,
)

ROOT = Path(__file__).resolve().parents[2]
START = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
FINAL = "lnsgkgsnl/1r5b1/pppppp1pp/6p2/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL b - 3"


@functools.cache
def built_oracle():
    subprocess.run(
        [
            "cargo",
            "build",
            "--locked",
            "-p",
            "open-shogi-core",
            "--example",
            "phase10u_replay",
            "--no-default-features",
            "--features",
            "pure-only",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        timeout=180,
    )
    return ROOT / "target/debug/examples/phase10u_replay"


@pytest.fixture
def evidence(tmp_path):
    oracle = built_oracle()
    if not oracle.is_file():
        pytest.fail("build rules oracle: cargo build -p open-shogi-core --example phase10u_replay")

    def store(name, data):
        (tmp_path / name).write_bytes(data)
        return {"path": name, "sha256": hashlib.sha256(data).hexdigest()}

    oracle_ref = store("oracle", oracle.read_bytes())
    (tmp_path / "oracle").chmod(0o755)
    model = Phase10TModel.random(20260907)
    identity = {
        "source_commit": "a" * 40,
        "cargo_features": ["pure-only"],
        "adapter_format": "OSAT10A1",
        "profile_name": "pure_learned",
        "evaluator_schema_sha256": "b" * 64,
        "native": store("native", b"pinned-native-fixture"),
        "wasm": None,
        "profile": store(
            "profile", (ROOT / "configs/runtime/pure_learned-a1-v1.json").read_bytes()
        ),
        "model": {
            **store("model", model.to_bytes()),
            "payload_sha256": hashlib.sha256(model.payload()).hexdigest(),
        },
    }
    identity["evaluator_schema_sha256"] = identity["profile"]["sha256"]
    identity["build_audit"] = store(
        "audit",
        canonical(
            {
                **{
                    key: identity[key]
                    for key in (
                        "source_commit",
                        "cargo_features",
                        "adapter_format",
                        "native",
                        "wasm",
                        "evaluator_schema_sha256",
                    )
                },
                "status": "PASS",
                "prohibited_modules_absent": True,
            }
        ),
    )
    runtime = {**dict.fromkeys(PROHIBITED, 0), "learned_eval_calls": 1}
    controls = {
        "warmup": "none; model startup excluded for both sides; all request setup charged",
        "clock": "monotonic_ns",
        "threads": 1,
        "hash_mb": 32,
        "search_options": {
            "configuration": "SearchConfig::default",
            "hash_mb": 32,
            "fresh_engine_per_move": True,
            "book": False,
        },
        "movetime_ns": 100_000_000,
        "hard_timeout_ns": 1_000_000_000,
        "nodes": None,
        "max_plies": 2,
        "max_depth": 64,
    }
    game = {
        "initial_sfen": START,
        "seed": 1,
        "pairing_id": "pair-1",
        "sides": {"black": identity, "white": identity},
        "controls": controls,
    }
    trusted = {
        "games": {"fixture": game},
        "replay_oracle": oracle_ref,
        "start_manifest": store("starts", canonical([{"sfen": START}])),
    }
    searches = [
        {
            "side": side,
            "move": move,
            "runtime": runtime,
            "engine_timing": {
                "clock": "monotonic",
                "scope": "request_parse_replay_setup_search",
                "requested_movetime_ms": 100,
                "hard_timeout_ms": 1000,
                "setup_elapsed_ns": 2,
                "search_start_ns": 2,
                "search_budget_ns": 99_999_998,
                "search_elapsed_ns": 5,
                "elapsed_ns": 8,
                "soft_budget_ns": 100_000_000,
                "hard_budget_ns": 1_000_000_000,
                "soft_compliant": True,
                "hard_compliant": True,
                "compliant": True,
            },
            "timing": {
                "clock": "monotonic_ns",
                "start_ns": index * 100,
                "end_ns": index * 100 + 10,
                "deadline_ns": index * 100 + 1_000_000_000,
                "elapsed_ns": 10,
                "compliant": True,
            },
        }
        for index, (side, move) in enumerate((("black", "7g7f"), ("white", "3c3d")))
    ]
    import subprocess

    for index, search in enumerate(searches):
        request = {
            "initial_sfen": START,
            "moves": ["7g7f", "3c3d"][:index],
            "depth": 64,
            "nodes": None,
            "movetime_ms": 100,
            "hard_timeout_ms": 1000,
            "hash_mb": 32,
        }
        pos = json.loads(
            subprocess.check_output(
                [str(oracle)],
                input=canonical({"initial_sfen": START, "moves": request["moves"], "max_plies": 2}),
            )
        )
        search["request"] = request
        search["response"] = {
            "requested_controls": request,
            "timing": search["engine_timing"],
            "best_move": search["move"],
            "model_format": "OSAT10A1",
            "model_sha256": identity["model"]["sha256"],
            "threads": 1,
            "hash_mb": 32,
            "depth_limit": 64,
            "node_limit": None,
            "proof": {
                **runtime,
                "model_sha256": identity["model"]["sha256"],
                "evaluator_profile_schema_hash": identity["profile"]["sha256"],
            },
            "final_sfen": pos["final_sfen"],
        }
    receipt = seal(
        {
            **game,
            "schema": SCHEMA,
            "game_id": "fixture",
            "manifest_sha256": digest(trusted),
            "start_manifest_sha256": trusted["start_manifest"]["sha256"],
            "runtime": {"black": runtime, "white": runtime},
            "searches": searches,
            "moves": ["7g7f", "3c3d"],
            "final_sfen": FINAL,
            "result": "excluded",
            "termination": "max_plies",
            "max_plies_reached": True,
        }
    )
    return tmp_path, receipt, trusted


def test_real_rules_replay_and_deterministic_immutable_receipt(evidence):
    root, receipt, trusted = evidence
    assert validate(root, receipt, trusted, digest(trusted))["result"] == "excluded"
    assert seal(receipt) == receipt
    assert canonical(receipt) == canonical(dict(reversed(list(receipt.items()))))
    output = root / "receipt.json"
    write_immutable(output, receipt)
    assert read_receipt(output) == receipt
    output.write_bytes(json.dumps(receipt).encode())
    with pytest.raises(EvidenceError):
        read_receipt(output)
    with pytest.raises(FileExistsError):
        write_immutable(output, receipt)


@pytest.mark.parametrize("field", ["native", "model", "profile"])
def test_mutated_artifact_rejected(evidence, field):
    root, receipt, trusted = evidence
    (root / field).write_bytes(b"tampered")
    with pytest.raises(EvidenceError):
        validate(root, receipt, trusted, digest(trusted))


@pytest.mark.parametrize(
    "kind",
    [
        "move",
        "binary_hash",
        "model_hash",
        "deadline",
        "result",
        "receipt",
        "missing",
        "runtime",
        "payload",
        "incomplete",
    ],
)
def test_tampering_rejected_even_when_resealed(evidence, kind):
    root, original, trusted = evidence
    receipt = copy.deepcopy(original)
    if kind == "move":
        receipt["moves"][0] = "7g7e"
        receipt["searches"][0]["move"] = "7g7e"
    elif kind in ("binary_hash", "model_hash"):
        key = "native" if kind == "binary_hash" else "model"
        receipt["sides"]["black"][key]["sha256"] = "0" * 64
    elif kind == "deadline":
        receipt["searches"][0]["timing"]["deadline_ns"] += 1
    elif kind == "result":
        receipt["result"] = "black"
    elif kind == "missing":
        del receipt["searches"][0]["timing"]["deadline_ns"]
    elif kind == "runtime":
        receipt["searches"][0]["runtime"]["fallback_count"] = 1
    elif kind == "payload":
        receipt["sides"]["black"]["model"]["payload_sha256"] = "0" * 64
    elif kind == "incomplete":
        receipt["moves"] = receipt["moves"][:1]
        receipt["searches"] = receipt["searches"][:1]
    receipt = seal(receipt)
    if kind == "receipt":
        receipt["receipt_sha256"] = "0" * 64
    with pytest.raises(EvidenceError):
        validate(root, receipt, trusted, digest(trusted))


def test_monotonic_measurement(monkeypatch):
    ticks = iter([100, 130])
    monkeypatch.setattr("time.monotonic_ns", lambda: next(ticks))
    assert MonotonicSearch(20).finish() == {
        "clock": "monotonic_ns",
        "start_ns": 100,
        "end_ns": 130,
        "deadline_ns": 120,
        "elapsed_ns": 30,
        "compliant": False,
    }


@pytest.mark.parametrize("field", ["seed", "threads", "max_plies_reached"])
def test_boolean_integer_confusion_rejected(evidence, field):
    root, receipt, trusted = evidence
    if field == "threads":
        receipt["controls"]["threads"] = True
    else:
        receipt[field] = True if field == "seed" else 1
    with pytest.raises(EvidenceError):
        validate(root, seal(receipt), trusted, digest(trusted))


@pytest.mark.parametrize("field", ["soft_budget_ns", "search_budget_ns", "hard_budget_ns"])
def test_engine_deadline_tamper_rejected(evidence, field):
    root, receipt, trusted = evidence
    receipt["searches"][0]["engine_timing"][field] += 1
    with pytest.raises(EvidenceError):
        validate(root, seal(receipt), trusted, digest(trusted))


def test_soft_setup_overrun_requires_explicit_hard_slack(evidence):
    root, receipt, trusted = evidence
    search = receipt["searches"][0]
    engine = search["engine_timing"]
    engine.update(
        {
            "setup_elapsed_ns": 100_000_001,
            "search_start_ns": 100_000_001,
            "search_budget_ns": 899_999_999,
            "search_elapsed_ns": 5,
            "elapsed_ns": 100_000_006,
            "soft_compliant": False,
        }
    )
    search["response"]["timing"] = engine
    with pytest.raises(EvidenceError, match="soft setup-overrun"):
        validate(root, seal(receipt), trusted, digest(trusted))
    engine["soft_budget_exhausted_in_setup"] = True
    engine["hard_deadline_safety_margin_ns"] = 5_000_000
    engine["search_budget_ns"] = 894_999_999
    # The outer coordinator timing is independent, but must still cover the engine exchange.
    search["timing"].update(
        {
            "start_ns": 0,
            "end_ns": 100_000_006,
            "deadline_ns": 1_000_000_000,
            "elapsed_ns": 100_000_006,
        }
    )
    second = receipt["searches"][1]
    second["timing"].update(
        {
            "start_ns": 100_000_100,
            "end_ns": 100_000_110,
            "deadline_ns": 1_100_000_100,
            "elapsed_ns": 10,
        }
    )
    assert validate(root, seal(receipt), trusted, digest(trusted))["result"] == "excluded"


def test_build_audit_rejects_uncertified_binary(evidence):
    root, receipt, trusted = evidence
    (root / "audit").write_text('{"status":"FAIL"}')
    with pytest.raises(EvidenceError):
        validate(root, receipt, trusted, digest(trusted))


@pytest.mark.parametrize(
    ("initial", "moves", "termination", "result"),
    [
        (START, ["5i6h", "5a6b", "6h5i", "6b5a"] * 3, "repetition", "draw"),
        (
            "4k4/5R3/9/9/9/9/9/9/K8 b - 1",
            ["4b5b", "5a4a", "5b4b", "4a5a"] * 3,
            "perpetual_check",
            "white",
        ),
        ("3lkl3/3pRp3/4G4/9/9/9/9/9/K8 w - 1", [], "checkmate", "black"),
        (START, [], "ongoing", "ongoing"),
    ],
)
def test_rules_oracle_terminal_classification(initial, moves, termination, result):
    import subprocess

    oracle = built_oracle()
    replay = json.loads(
        subprocess.check_output(
            [str(oracle)],
            input=canonical({"initial_sfen": initial, "moves": moves, "max_plies": 100}),
        )
    )
    assert replay["termination"] == termination
    assert replay["result"] == result


def test_rules_oracle_rejects_post_terminal_move():
    import subprocess

    oracle = built_oracle()
    result = subprocess.run(
        [str(oracle)],
        input=canonical(
            {
                "initial_sfen": START,
                "moves": ["5i6h", "5a6b", "6h5i", "6b5a"] * 3 + ["7g7f"],
                "max_plies": 100,
            }
        ),
        capture_output=True,
    )
    assert result.returncode != 0


def test_separately_certified_handcrafted_opponent(evidence):
    root, receipt, trusted = evidence
    identity = copy.deepcopy(receipt["sides"]["white"])
    profile = canonical({"profile": "handcrafted_experimental"})
    (root / "handcrafted-profile").write_bytes(profile)
    identity.update(
        {
            "adapter_format": "HANDCRAFTED",
            "profile_name": "handcrafted_experimental",
            "cargo_features": ["handcrafted"],
            "model": None,
            "profile": {
                "path": "handcrafted-profile",
                "sha256": hashlib.sha256(profile).hexdigest(),
            },
            "evaluator_schema_sha256": hashlib.sha256(profile).hexdigest(),
        }
    )
    audit = canonical(
        {
            **{
                key: identity[key]
                for key in (
                    "source_commit",
                    "cargo_features",
                    "adapter_format",
                    "native",
                    "wasm",
                    "evaluator_schema_sha256",
                )
            },
            "status": "PASS",
        }
    )
    (root / "handcrafted-audit").write_bytes(audit)
    identity["build_audit"] = {
        "path": "handcrafted-audit",
        "sha256": hashlib.sha256(audit).hexdigest(),
    }
    trusted["games"]["fixture"]["sides"]["white"] = identity
    receipt["sides"]["white"] = identity
    receipt["manifest_sha256"] = digest(trusted)
    counters = {
        **dict.fromkeys(PROHIBITED, 0),
        "handcrafted_eval_calls": 1,
        "learned_eval_calls": 0,
    }
    receipt["runtime"]["white"] = counters
    receipt["searches"][1]["runtime"] = counters
    response = receipt["searches"][1]["response"]
    response.update({"model_format": "HANDCRAFTED", "model_sha256": "", "proof": counters})
    assert validate(root, seal(receipt), trusted, digest(trusted))["result"] == "excluded"
