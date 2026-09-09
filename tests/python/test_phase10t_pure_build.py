from __future__ import annotations

import hashlib

import pytest
from open_shogi_training.phase10t_pure_build import (
    FORBIDDEN_COUNTERS,
    portable_text,
    quantize_cp,
    scan_prohibited,
    verify_counters,
    verify_evidence,
    verify_parity,
    wasm_sections,
)


def test_counter_evidence_fails_closed() -> None:
    proof = dict.fromkeys(FORBIDDEN_COUNTERS, 0) | {"learned_eval_calls": 1}
    assert verify_counters(proof) == proof
    for key in (*FORBIDDEN_COUNTERS, "learned_eval_calls"):
        missing = proof.copy()
        del missing[key]
        with pytest.raises(ValueError):
            verify_counters(missing)
        with pytest.raises(ValueError):
            verify_counters(proof | {key: True})
    for key in FORBIDDEN_COUNTERS:
        with pytest.raises(ValueError):
            verify_counters(proof | {key: 1})


def test_artifact_scan_checks_code_and_data_not_only_symbols() -> None:
    header = b"\0asm\x01\0\0\0"
    assert wasm_sections(header + b"\x0a\x01\x00")[0]["id"] == 10
    marker = b"NeuralEvaluator"
    with pytest.raises(ValueError, match="prohibited"):
        wasm_sections(header + b"\x0a\x01\x00\x0b" + bytes([len(marker)]) + marker)
    with pytest.raises(ValueError):
        scan_prohibited(b"open_shogi_core::evaluation::evaluate", "native")
    for data in (b"", header, header + b"\x0a\x04\x00", header + b"\x0a\x80"):
        with pytest.raises(ValueError):
            wasm_sections(data)


def test_parity_rejects_nan_and_wrong_score() -> None:
    expected = {"cp": 0, "wdl_logits": [0.1, 0.2, 0.3]}
    verify_parity(expected, expected)
    with pytest.raises(ValueError):
        verify_parity(expected, expected | {"cp": 1})
    with pytest.raises(ValueError):
        verify_parity(expected, expected | {"wdl_logits": [float("nan"), 0.2, 0.3]})


def test_evidence_rehashes_runtime_sources_and_artifacts(tmp_path, monkeypatch) -> None:
    import open_shogi_training.phase10t_pure_build as module

    path = tmp_path / "core.rs"
    path.write_bytes(b"source")
    sha = hashlib.sha256(b"source").hexdigest()
    monkeypatch.setattr(module, "source_inventory", lambda root: {"core.rs": module.digest(path)})
    proof = dict.fromkeys(FORBIDDEN_COUNTERS, 0) | {"learned_eval_calls": 1}
    profile = tmp_path / "configs/runtime/pure_learned-a1-v1.json"
    profile.parent.mkdir(parents=True)
    profile.write_bytes(b"source")
    proof.update(
        {
            "model_sha256": sha,
            "evaluator_profile_schema_hash": sha,
            "profile": "pure_learned",
            "profile_schema": "open_shogiai_pure_learned_a1_profile/v1",
        }
    )
    reference = {
        "cp": 0,
        "wdl_logits": [0.1, 0.2, 0.3],
        "compiled_evaluators": ["osaval02", "phase10t-a1"],
    }
    wasm_file = tmp_path / "engine.wasm"
    wasm_file.write_bytes(b"\0asm\x01\0\0\0\x0a\x01\x00")
    (tmp_path / "native.map").write_bytes(b"source")
    generated = {}
    for directory in ("wasm-one", "wasm-two"):
        folder = tmp_path / directory
        folder.mkdir()
        for name in (
            "open_shogi_wasm.js",
            "open_shogi_wasm.d.ts",
            "open_shogi_wasm_bg.wasm",
            "open_shogi_wasm_bg.wasm.d.ts",
        ):
            (folder / name).write_bytes(b"source")
            generated[name] = sha
    (tmp_path / "metadata.log").write_text("{}")
    receipt = {
        "schema": "open_shogiai_phase10t_pure_artifact_audit/v1",
        "status": "passed",
        "git_status": "",
        "git_commit": "a" * 40,
        "source_hashes": {"core.rs": sha},
        "artifacts": {
            key: {"path": "core.rs", "sha256": sha}
            for key in ("native", "wasm_raw", "wasm", "model")
        },
        "negative_imports": module.NEGATIVE_IMPORTS,
        "evaluator_profile_schema_hash": sha,
        "cargo_features": ["pure-only"],
        "compiled_evaluator_registrations": ["osaval02", "phase10t-a1"],
        "source_depfiles": {"core.rs": sha},
        "output_directory": ".",
        "native_link_map_sha256": sha,
        "deterministic_wasm_regeneration": [generated, generated],
        "graphs": {"open-shogi-cli": "source", "open-shogi-wasm": "source"},
        "metadata": {},
        "rustc_cfg": "source",
        "commands": [
            {"argv": ["cargo", "tree", "-p", "open-shogi-cli"], "log": "core.rs", "sha256": sha},
            {"argv": ["cargo", "tree", "-p", "open-shogi-wasm"], "log": "core.rs", "sha256": sha},
            {"argv": ["cargo", "rustc", "--print", "cfg"], "log": "core.rs", "sha256": sha},
            {
                "argv": ["cargo", "metadata"],
                "log": "metadata.log",
                "sha256": module.digest(tmp_path / "metadata.log"),
            },
        ],
        "wasm_raw_sections": module.wasm_sections(wasm_file.read_bytes()),
        "wasm_sections": module.wasm_sections(wasm_file.read_bytes()),
        "runtime": {
            "model_sha256": sha,
            "negative_model_cases": module.NEGATIVE_CASES,
            "native": [reference | {"proof": proof}] * 10,
            "python": [reference] * 10,
            "invalid_histories": [{}] * 4,
            "wasm": {
                "invalidHistoryCount": 4,
                "failures": [case for case in module.NEGATIVE_CASES if case != "missing"],
                "outputs": [{"evaluate": reference, "search": {"proof": proof}}] * 10,
            },
            "usi_proofs": [proof],
        },
    }
    for key in ("wasm", "wasm_raw"):
        receipt["artifacts"][key] = {"path": "engine.wasm", "sha256": module.digest(wasm_file)}
    verify_evidence(tmp_path, receipt)
    with pytest.raises(ValueError, match="clean committed"):
        verify_evidence(tmp_path, receipt | {"git_status": " M engine/core/src/lib.rs"})
    with pytest.raises(ValueError, match="clean committed"):
        verify_evidence(tmp_path, receipt | {"git_commit": "A" * 40})
    with pytest.raises(ValueError):
        verify_evidence(tmp_path, receipt | {"artifacts": {}})
    with pytest.raises(ValueError):
        verify_evidence(tmp_path, receipt | {"source_hashes": {}})
    (tmp_path / "native.map").write_bytes(b"changed")
    with pytest.raises(ValueError, match="link map changed"):
        verify_evidence(tmp_path, receipt)
    (tmp_path / "native.map").write_bytes(b"source")
    (tmp_path / "wasm-two/open_shogi_wasm.js").write_bytes(b"changed")
    with pytest.raises(ValueError, match="generated Wasm files changed"):
        verify_evidence(tmp_path, receipt)
    (tmp_path / "wasm-two/open_shogi_wasm.js").write_bytes(b"source")
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source changed"):
        verify_evidence(tmp_path, receipt)


def test_cp_semantics_round_away_from_zero_and_exclude_mate_range() -> None:
    assert [quantize_cp(x) for x in (-0.5, 0.5, -40000.0, 40000.0)] == [-1, 1, -28999, 28999]


def test_portable_metadata_paths_preserve_repository_precedence(monkeypatch) -> None:
    from pathlib import Path

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/fixture/home")))
    root = Path("/fixture/home/project")
    text = (
        '{"manifest":"/fixture/home/project/Cargo.toml",'
        '"registry":"/fixture/home/.cargo/registry/package"}'
    )
    assert portable_text(text, root) == (
        '{"manifest":"${REPO}/Cargo.toml","registry":"${HOME}/.cargo/registry/package"}'
    )

    uri = "file" + "://" + str(root) + "/Cargo.toml"
    assert portable_text(uri, root) == "${REPO_URI}/Cargo.toml"
