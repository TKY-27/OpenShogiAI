import hashlib
import json
from pathlib import Path

from open_shogi_training.phase10v_build_compat import ALLOWED, BASE, runtime_successor_matches


def test_explicit_successor_requires_exact_prior_and_current_hash(tmp_path, monkeypatch):
    before = hashlib.sha256(b"preserved").hexdigest()
    current = hashlib.sha256(b"successor").hexdigest()
    legacy = hashlib.sha256(b"old").hexdigest()
    directory = tmp_path / "configs/phase10v"
    directory.mkdir(parents=True)
    manifest = {
        "schema": "open_shogiai_phase10v_runtime_successors/v1",
        "base_commit": BASE,
        "files": {p: {"before": before, "after": current} for p in ALLOWED},
    }
    (directory / "runtime-successors.json").write_text(json.dumps(manifest))
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: None)
    monkeypatch.setattr("subprocess.check_output", lambda *a, **kw: b"preserved")
    prior = {"before": legacy, "after": before}
    name = "engine/core/src/lib.rs"
    assert runtime_successor_matches(tmp_path, name, legacy, current, prior)
    assert not runtime_successor_matches(tmp_path, name, legacy, "a" * 64, prior)
    assert not runtime_successor_matches(tmp_path, name, "b" * 64, current, prior)
    assert not runtime_successor_matches(
        tmp_path, "configs/phase10r/targets.json", legacy, current, prior
    )
    monkeypatch.setattr("subprocess.check_output", lambda *a, **kw: b"different baseline")
    assert not runtime_successor_matches(tmp_path, name, legacy, current, prior)


def test_current_chain_retains_phase10u_baseline():
    from open_shogi_training.phase10t_build_compat import runtime_successor_matches as matches

    root = Path(__file__).resolve().parents[2]
    prior = json.loads((root / "configs/phase10t/runtime-successors.json").read_text())
    for name in ALLOWED:
        assert matches(
            root,
            name,
            prior["files"][name]["before"],
            hashlib.sha256((root / name).read_bytes()).hexdigest(),
        )
