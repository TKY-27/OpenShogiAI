import pytest
from open_shogi_training.phase10t_pure_build import FORBIDDEN_COUNTERS
from open_shogi_training.phase10u_prior_audit import close_tree, model_identity, verify_identity


def test_prior_proof_binds_model_profile_and_counters():
    identity = {"sha256": "a" * 64, "profile_sha256": "b" * 64, "profile_schema": "legacy/v1"}
    proof = dict.fromkeys(FORBIDDEN_COUNTERS, 0) | {
        "learned_eval_calls": 1,
        "profile": "pure_learned",
        "model_sha256": identity["sha256"],
        "evaluator_profile_schema_hash": identity["profile_sha256"],
        "profile_schema": identity["profile_schema"],
    }
    verify_identity(proof, identity)
    for key in proof:
        with pytest.raises(ValueError):
            verify_identity({k: v for k, v in proof.items() if k != key}, identity)
    for key in ("profile", "model_sha256", "evaluator_profile_schema_hash", "profile_schema"):
        with pytest.raises(ValueError):
            verify_identity(proof | {key: "changed"}, identity)


def test_full_inference_parity_rejects_tampering():
    expected = {"cp": 42, "logits": [0.1, 0.2], "identity": {"format": "OSAVAL02"}}
    close_tree(expected, expected)
    for changed in (
        expected | {"cp": 41},
        expected | {"logits": [float("nan"), 0.2]},
        expected | {"logits": [0.1]},
        expected | {"extra": 0},
        expected | {"identity": {"format": "OSAT10A1"}},
    ):
        with pytest.raises(ValueError):
            close_tree(expected, changed)


def test_exact_frozen_model_identity_required(tmp_path):
    (tmp_path / "model.bin").write_bytes(b"OSAVAL02")
    with pytest.raises(ValueError, match="identity mismatch"):
        model_identity(
            tmp_path, {"id": "prior-1m", "model": {"path": "model.bin", "sha256": "0" * 64}}
        )
    with pytest.raises(ValueError, match="repository relative"):
        model_identity(
            tmp_path, {"id": "prior-1m", "model": {"path": "../model.bin", "sha256": "0" * 64}}
        )


def test_player_deadline_and_request_binding():
    from open_shogi_training.phase10u_prior_audit import verify_player

    identity = {
        "sha256": "a" * 64,
        "profile_sha256": "b" * 64,
        "profile_schema": "legacy/v1",
        "format": "OSAVAL02",
    }
    request = {"hard_timeout_ms": 30}
    ready = {
        "ready": True,
        "model_format": "OSAVAL02",
        "model_sha256": identity["sha256"],
        "compiled_evaluators": ["osaval02", "phase10t-a1"],
    }
    timing = {
        "clock": "monotonic",
        "hard_compliant": True,
        "hard_timeout_ms": 30,
        "hard_budget_ns": 30_000_000,
        "setup_elapsed_ns": 10,
        "search_start_ns": 11,
        "search_elapsed_ns": 20,
        "elapsed_ns": 31,
    }
    response = {
        "requested_controls": request,
        "best_move": "7g7f",
        "model_format": "OSAVAL02",
        "model_sha256": identity["sha256"],
        "threads": 1,
        "hash_mb": 32,
        "deadline": timing,
        "proof": dict.fromkeys(FORBIDDEN_COUNTERS, 0)
        | {
            "learned_eval_calls": 1,
            "profile": "pure_learned",
            "model_sha256": identity["sha256"],
            "evaluator_profile_schema_hash": identity["profile_sha256"],
            "profile_schema": "legacy/v1",
        },
    }
    verify_player([ready, response], request, identity)
    for changed in (
        timing | {"clock": "wall"},
        timing | {"hard_compliant": False},
        timing | {"elapsed_ns": 30_000_001},
        timing | {"elapsed_ns": 29},
        timing | {"hard_timeout_ms": 31},
    ):
        with pytest.raises(ValueError, match="deadline"):
            verify_player([ready, response | {"deadline": changed}], request, identity)
    with pytest.raises(ValueError, match="transport"):
        verify_player(
            [ready, response | {"requested_controls": {"hard_timeout_ms": 31}}], request, identity
        )
