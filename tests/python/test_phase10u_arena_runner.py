"""Bounded fake transports exercise coordinator gates without model play or Arena games."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import pytest
from open_shogi_training import phase10u_arena_evidence as evidence
from open_shogi_training import phase10u_arena_runner as runner


def fake_player(
    root: Path, *, ready_override: dict | None = None, wait: bool = False, no_read: bool = False
) -> dict:
    ready = {
        "schema": "open_shogiai_phase10u_arena_player_ready/v1",
        "profile": "experimental",
        "adapter_format": "HANDCRAFTED",
        "model_sha256": None,
    }
    ready.update(ready_override or {})
    script = (
        f"#!{sys.executable}\n"
        "import json,sys,time\n"
        f"print({json.dumps(json.dumps(ready))}, flush=True)\n"
        + ("time.sleep(5)\n" if no_read else "")
        + "for line in sys.stdin:\n"
        + ("    time.sleep(5)\n" if wait else "    print('{}', flush=True)\n")
    ).encode()
    sha = hashlib.sha256(script).hexdigest()
    path = root / "players" / sha / "player"
    path.parent.mkdir(parents=True)
    path.write_bytes(script)
    path.chmod(0o755)
    return {
        "native": {"path": path.relative_to(root).as_posix(), "sha256": sha},
        "profile_name": "experimental",
        "adapter_format": "HANDCRAFTED",
        "model": None,
    }


def test_content_addressed_startup_then_mutated_native_is_rejected(tmp_path: Path) -> None:
    identity = fake_player(tmp_path)
    player = runner.Player(tmp_path, identity, tmp_path / "player.log")
    try:
        assert player.ready["adapter_format"] == "HANDCRAFTED"
        native = tmp_path / identity["native"]["path"]
        native.write_bytes(native.read_bytes() + b"# changed after startup\n")
        with pytest.raises(evidence.EvidenceError, match="SHA-256"):
            player.search({"hard_timeout_ms": 50})
    finally:
        player.close()
    assert player.process.poll() is not None
    assert b"arena_player_ready" in (tmp_path / "player.log").read_bytes()


@pytest.mark.parametrize(
    "override",
    [
        {"profile": "pure-learned"},
        {"adapter_format": "OSAVAL02"},
        {"model_sha256": "0" * 64},
        {"schema": "wrong-schema"},
    ],
)
def test_startup_identity_mismatch_retains_observed_response(
    tmp_path: Path, override: dict
) -> None:
    identity = fake_player(tmp_path, ready_override=override)
    log = tmp_path / "bad.log"
    with pytest.raises(evidence.EvidenceError, match="startup identity"):
        runner.Player(tmp_path, identity, log)
    assert json.loads(log.read_text()).items() >= override.items()


def test_mutable_target_path_cannot_launch_even_with_matching_digest(tmp_path: Path) -> None:
    identity = fake_player(tmp_path)
    original = tmp_path / identity["native"]["path"]
    mutable = tmp_path / "target-player"
    mutable.write_bytes(original.read_bytes())
    mutable.chmod(0o755)
    identity["native"]["path"] = "target-player"
    with pytest.raises(evidence.EvidenceError, match="content-addressed"):
        runner.Player(tmp_path, identity, tmp_path / "not-created.log")
    assert not (tmp_path / "not-created.log").exists()


def test_unresponsive_transport_has_bounded_per_search_deadline(tmp_path: Path) -> None:
    identity = fake_player(tmp_path, wait=True)
    player = runner.Player(tmp_path, identity, tmp_path / "deadline.log")
    started = time.monotonic()
    try:
        with pytest.raises(evidence.EvidenceError, match="deadline"):
            player.search({"hard_timeout_ms": 20})
        assert time.monotonic() - started < 2
    finally:
        player.close()
    assert player.process.poll() is not None


@pytest.mark.parametrize("authorization", [{}, {"execution_authorized": False}])
def test_closed_authorization_starts_nothing_and_creates_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, authorization: dict
) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("closed authorization must not reach player launch")

    monkeypatch.setattr(runner, "Player", forbidden)
    manifest = {"games": {}}
    with pytest.raises(evidence.EvidenceError, match="authorization"):
        runner.run(tmp_path, manifest, evidence.digest(manifest), authorization, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_failed_attempt_retained_and_cannot_be_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Unit-test retention after a passed preflight, without substituting fake trust for
    # the integration validator or executing a single game/search.
    if hasattr(evidence, "validate_manifest"):
        monkeypatch.setattr(evidence, "validate_manifest", lambda *args, **kwargs: None)
    identity = fake_player(tmp_path, ready_override={"adapter_format": "wrong"})
    manifest = {"games": {"fixture": {"sides": {"black": identity, "white": identity}}}}
    sha = evidence.digest(manifest)
    authorization = {
        "schema": "open_shogiai_phase10u_later_arena_authorization/v1",
        "manifest_sha256": sha,
        "execution_authorized": True,
    }
    out = tmp_path / "attempt"
    with pytest.raises(evidence.EvidenceError, match="startup identity"):
        runner.run(tmp_path, manifest, sha, authorization, out)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    assert "fixture-black.log" in before
    assert json.loads(before["STOP_CLOSED.json"])["status"] == "STOP_CLOSED"
    assert json.loads(before["manifest.json"]) == manifest
    assert not (out / "fixture.json").exists()
    with pytest.raises(FileExistsError):
        runner.run(tmp_path, manifest, sha, authorization, out)
    assert before == {p.name: p.read_bytes() for p in out.iterdir()}


def test_existing_transport_log_is_never_overwritten(tmp_path: Path) -> None:
    identity = fake_player(tmp_path)
    log = tmp_path / "existing.log"
    log.write_bytes(b"preserved failure evidence\n")
    with pytest.raises(FileExistsError):
        runner.Player(tmp_path, identity, log)
    assert log.read_bytes() == b"preserved failure evidence\n"


@pytest.mark.parametrize("not_boolean", [1, 1.0, "true", None])
def test_authorization_requires_literal_boolean_true(not_boolean: object) -> None:
    manifest = {"games": {}}
    sha = evidence.digest(manifest)
    authorization = {
        "schema": "open_shogiai_phase10u_later_arena_authorization/v1",
        "manifest_sha256": sha,
        "execution_authorized": not_boolean,
    }
    with pytest.raises(evidence.EvidenceError, match="authorization"):
        runner.require_authorization(manifest, sha, authorization)


def test_player_that_never_reads_cannot_block_request_writer(tmp_path: Path) -> None:
    identity = fake_player(tmp_path, no_read=True)
    player = runner.Player(tmp_path, identity, tmp_path / "blocked-write.log")
    started = time.monotonic()
    try:
        with pytest.raises(evidence.EvidenceError, match="request deadline"):
            player.send(b"x" * 2_000_000, time.monotonic_ns() + 20_000_000)
        assert time.monotonic() - started < 2
    finally:
        player.close()


def test_uncorrelated_response_is_rejected(tmp_path: Path) -> None:
    identity = fake_player(tmp_path)
    player = runner.Player(tmp_path, identity, tmp_path / "uncorrelated.log")
    try:
        with pytest.raises(evidence.EvidenceError, match="controls mismatch"):
            player.search({"hard_timeout_ms": 1000})
    finally:
        player.close()


def test_expired_deadline_rejects_even_buffered_response(tmp_path: Path) -> None:
    identity = fake_player(tmp_path)
    player = runner.Player(tmp_path, identity, tmp_path / "buffered.log")
    try:
        player.pending = b"{}\n"
        with pytest.raises(evidence.EvidenceError, match="deadline"):
            player.receive(time.monotonic_ns() - 1)
    finally:
        player.close()


@pytest.mark.parametrize("tamper_response", [False, True])
def test_coordinator_fixture_receipt_passes_independent_real_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper_response: bool
) -> None:
    import copy

    from test_phase10u_arena_evidence import evidence as fixture_factory

    # Reuse the evidence suite's bounded scripted transcript and real rules oracle.
    # These synthetic artifact identities certify no production player or Arena run.
    root, expected, trusted = fixture_factory.__wrapped__(tmp_path)
    created = []

    class ScriptedPlayer:
        def __init__(self, root, identity, log):
            self.side = "black" if log.name.endswith("-black.log") else "white"
            self.closed = False
            self.searched = False
            created.append(self)

        def search(self, request):
            assert not self.searched
            self.searched = True
            entry = next(item for item in expected["searches"] if item["side"] == self.side)
            assert evidence.canonical(request) == evidence.canonical(entry["request"])
            response = copy.deepcopy(entry["response"])
            if tamper_response and self.side == "black":
                response["model_sha256"] = "0" * 64
            return response, copy.deepcopy(entry["timing"])

        def close(self):
            self.closed = True

    monkeypatch.setattr(runner, "Player", ScriptedPlayer)
    output = tmp_path / "scripted-output"
    output.mkdir()
    args = (root, trusted, evidence.digest(trusted), "fixture", trusted["games"]["fixture"], output)
    if tamper_response:
        with pytest.raises(evidence.EvidenceError):
            runner._game(*args)
        assert not (output / "fixture.json").exists()
    else:
        runner._game(*args)
        actual = evidence.read_receipt(output / "fixture.json")
        assert evidence.canonical(actual) == evidence.canonical(expected)
        assert (
            evidence.validate(root, actual, trusted, evidence.digest(trusted))["result"]
            == "excluded"
        )
    assert len(created) == 2
    assert all(player.closed and player.searched for player in created)
