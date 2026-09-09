import json
from pathlib import Path

import pytest
from open_shogi_training.phase10t import sha256, tree_identity
from open_shogi_training.phase10v_audit import FORBIDDEN, check_proof
from open_shogi_training.phase10v_preservation import verify


def test_runtime_proof_rejects_wrong_model_profile_and_noninteger_counters():
    proof = {
        "model_sha256": "a" * 64,
        "learned_eval_calls": 1,
        "profile": "pure_learned",
        "profile_schema": "open_shogiai_pure_learned_v3_profile/v1",
        **dict.fromkeys(FORBIDDEN, 0),
    }
    result = {"proof": proof, "best_move": "7g7f", "nodes": 2}
    check_proof(result, "a" * 64)
    for change in (
        {"model_sha256": "b" * 64},
        {"profile": "standard"},
        {"learned_eval_calls": False},
        {"fallback_count": 1},
        {"teacher_calls": False},
    ):
        with pytest.raises(ValueError):
            check_proof({**result, "proof": {**proof, **change}}, "a" * 64)


def test_preservation_rechecks_bytes_and_tree_membership(tmp_path: Path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    model = evidence / "model.bin"
    model.write_bytes(b"immutable model")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "open_shogiai_phase10v_preservation/v1",
                "files": [
                    {
                        "path": "evidence/model.bin",
                        "bytes": model.stat().st_size,
                        "sha256": sha256(model),
                    }
                ],
                "trees": [tree_identity(tmp_path, "evidence")],
            }
        )
    )
    assert verify(tmp_path, manifest)["status"] == "PASS"
    (evidence / "new-evidence.json").write_text("{}")
    with pytest.raises(ValueError, match="tree changed"):
        verify(tmp_path, manifest)
    model.write_bytes(b"changed")
    with pytest.raises(ValueError, match="file changed"):
        verify(tmp_path, manifest)
