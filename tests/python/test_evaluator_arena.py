"""Light contracts only: no model training, external teacher, or real matches."""

from __future__ import annotations

import gzip
from pathlib import Path
from types import SimpleNamespace

import pytest
from open_shogi_training import evaluator_arena as arena
from open_shogi_training.evaluator_data import START, digest, encoded


def sfen(index: int, move: int = 17) -> str:
    ranks = ["4k4", "9", "9", "9", "9", "9", "9", "9", "K8"]
    file = index % 9
    ranks[2 + index // 9] = (str(file) if file else "") + "P" + (str(8 - file) if file < 8 else "")
    return f"{'/'.join(ranks)} b - {move}"


def dataset(folder: Path, count: int = 16) -> None:
    folder.mkdir(parents=True)
    rows = [{"game": i + 8, "ply": 16, "sfen": sfen(i), "kind": "root"} for i in range(count)]
    rows.append({"game": 8, "ply": 12, "sfen": sfen(18, 13), "kind": "root"})
    path = folder / "development_test-rows.jsonl.gz"
    path.write_bytes(gzip.compress(b"".join(encoded(r) + b"\n" for r in rows)))
    (folder / "manifest.json").write_bytes(
        encoded(
            {
                "source_games": [
                    {"game": i + 8, "split": "development_test"} for i in range(count)
                ],
                "artifacts": [{"path": path.name, "sha256": digest(path)}],
            }
        )
    )


def configuration(root: Path) -> dict:
    for name in ("probe", "baseline", "candidate", "scripts/compare_core_prototype.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    dataset(root / "local/dataset")
    return {
        "probe_path": "probe",
        "probe_sha256": digest(root / "probe"),
        "baseline_path": "baseline",
        "baseline_sha256": digest(root / "baseline"),
        "candidate_path": "candidate",
        "candidate_sha256": digest(root / "candidate"),
        "dataset_path": "local/dataset",
        "seed": 20260915,
        "depth": 64,
        "max_plies": 256,
    }


def test_start_selection_is_finite_distinct_paired_and_label_independent(tmp_path):
    config = configuration(tmp_path)
    plan = arena._plan(tmp_path, config)
    games = plan["games"]
    assert len(games) == 40
    assert sum(g["group"] == "evaluation" and g["clock_ms"] == 180_000 for g in games) == 24
    assert sum(g["group"] == "evaluation" and g["clock_ms"] == 600_000 for g in games) == 8
    assert len({g["source_game"] for g in games[:32]}) == 16
    assert all(g["source_ply"] == 16 for g in games[:32])
    for left, right in zip(games[::2], games[1::2], strict=True):
        assert left["initial_sfen"] == right["initial_sfen"]
        assert {left["candidate_side"], right["candidate_side"]} == {"black", "white"}
    assert games == arena._plan(tmp_path, config)["games"]
    assert plan["sealed_holdout_opened"] is False


def test_missing_unique_source_roots_fail_instead_of_recycling(tmp_path):
    dataset(tmp_path / "data", count=15)
    with pytest.raises(ValueError, match="16 distinct"):
        arena._starts(tmp_path / "data", 20260915)


def response(model: str, *, nodes: int = 1, terminal: bool = False) -> dict:
    calls = 0 if nodes == 0 else 1
    return {
        "schema": "open_shogiai_core_probe/v1",
        "leaf_sha256": model,
        "proof": {
            "profile": "pure_learned",
            "model_sha256": model,
            "profile_schema": "open_shogiai_pure_learned_v3_profile/v1",
            "evaluator_profile_schema_hash": arena.PROFILE_HASH,
            "learned_eval_calls": calls,
            "accumulator_updates": calls,
            "accumulator_refreshes": calls,
            **dict.fromkeys(arena.FORBIDDEN, 0),
        },
        "nodes": nodes,
        "depth": int(nodes > 0),
        "seldepth": int(nodes > 0),
        "score": -30000 if terminal else 0,
        "compute_control": None,
        "outcome": "checkmate"
        if terminal
        else "evaluated"
        if nodes
        else "node_limit_before_evaluation",
        "termination": "Completed" if terminal else "Stable" if nodes else "NodeLimit",
        "best_move": None if terminal else "7g7f",
        "legal": not terminal,
        "game_end": "Some(Checkmate { winner: Black })" if terminal else "None",
    }


def test_runtime_proof_requires_positive_normal_work_but_allows_declared_exceptions():
    for document in (response("a"), response("a", nodes=0), response("a", nodes=0, terminal=True)):
        arena._validate_response(document, "a")
    malformed = response("a")
    malformed["proof"]["learned_eval_calls"] = 0
    with pytest.raises(ValueError, match="positive"):
        arena._validate_response(malformed, "a")
    forbidden = response("a")
    forbidden["proof"]["teacher_calls"] = 1
    with pytest.raises(ValueError, match="isolation"):
        arena._validate_response(forbidden, "a")
    with pytest.raises(ValueError, match="identity"):
        arena._validate_response(response("old"), "new")


def test_clock_ceiling_uses_sfen_move_number_and_conservative_milliseconds():
    assert arena._clock_limits(START, 600_000_999_999) == (18_000, 17_900)
    assert arena._clock_limits(START, 180_000_000_000) == (5_400, 5_300)
    assert arena._clock_limits(sfen(0), 600_000_000_000) == (19_563, 19_463)
    assert arena._clock_limits(START, 999_999) == (0, 0)


def completed(plan: dict, value: float = 1.0) -> list[dict]:
    return [
        {
            **g,
            "status": "completed",
            "score_candidate": value,
            "reason": "checkmate",
            "absolute_deadline_violations": 0,
        }
        for g in plan["games"]
    ]


def test_acceptance_keeps_losses_incomplete_and_recovered_failures_visible(tmp_path):
    plan = arena._plan(tmp_path, configuration(tmp_path))
    wins = completed(plan)
    assert arena._summary(plan, wins, wins)["adoption_criteria_met"]
    draws = completed(plan, 0.5)
    assert not arena._summary(plan, draws, draws)["adoption_criteria_met"]
    missing = wins[:-1]
    assert not arena._summary(plan, missing, missing)["adoption_criteria_met"]
    adverse = [*wins, {"status": "process_failure", "reason": "crash"}]
    assert not arena._summary(plan, wins, adverse)["adoption_criteria_met"]
    one_clock_loses = [
        dict(g, score_candidate=0.0) if g["clock_ms"] == 600_000 else g for g in wins
    ]
    assert not arena._summary(plan, one_clock_loses, one_clock_loses)["adoption_criteria_met"]
    assert arena._summary(plan, wins, wins)["promotion_performed"] is False


class SimulatedClock:
    def __init__(self):
        self.now = 1_000_000_000

    def advance(self, ms):
        self.now += ms * arena.NS_PER_MS


def helper(root, clock):
    class ProbeError(Exception):
        pass

    class Probe:
        def __init__(self, argv, name, folder):
            self.model = argv[2]
            self.name = name
            self.stderr = folder / f"{name}.stderr.log"
            self.stderr.write_bytes(b"")

        def exchange(self, request, timeout_ns):
            assert timeout_ns > 0
            terminal = bool(request.get("moves")) and request.get("nodes") == 0
            document = response(self.model, nodes=0 if "nodes" in request else 1, terminal=terminal)
            document["sfen"] = (
                request["sfen"] if not terminal else request["sfen"].replace(" b - 1", " w - 2")
            )
            document["perspective"] = "White" if terminal else "Black"
            if "black_time_ms" in request:
                _, hard = arena._clock_limits(
                    request["sfen"], request["black_time_ms"] * arena.NS_PER_MS
                )
                document["time_hard_limit_ms"] = hard
                clock.advance(7)
            elif terminal:
                clock.advance(13)
            else:
                clock.advance(50)
            return {"response": document}

        def close(self):
            return {
                "returncode": 0,
                "shutdown": "stdin_eof",
                "stderr_path": str(self.stderr.relative_to(root)),
                "stderr_sha256": digest(self.stderr),
            }

    class Trace:
        def __init__(self, path):
            self.path, self.data = path, bytearray()

        def write(self, data):
            self.data.extend(data)

        def flush(self):
            pass

        def close(self):
            self.path.write_bytes(gzip.compress(bytes(self.data)))
            return {"path": str(self.path.relative_to(root)), "sha256": digest(self.path)}

    return SimpleNamespace(
        Probe=Probe,
        ProbeError=ProbeError,
        TraceJournal=Trace,
        canonical=lambda x: encoded(x) + b"\n",
        cpu_usage=lambda: {},
        cpu_delta=lambda *_: {},
        opposite=lambda side: "white" if side == "black" else "black",
        adjudicated=lambda r: (
            None if r["game_end"] == "None" else {"winner": "black", "reason": "checkmate"}
        ),
    )


def test_move_application_time_is_charged_and_receipt_binds_trace(tmp_path, monkeypatch):
    plan = arena._plan(tmp_path, configuration(tmp_path))
    game = plan["games"][32]
    output = tmp_path / "local/arena"
    folder = output / "games/test/attempt-000"
    folder.mkdir(parents=True)
    clock = SimulatedClock()
    monkeypatch.setattr(arena.time, "monotonic_ns", lambda: clock.now)
    monkeypatch.setattr(arena.time, "monotonic", lambda: clock.now / 1e9)
    result = arena._play_game(
        tmp_path, output, game, plan, "plan-hash", folder, helper(tmp_path, clock), None, 10_000
    )
    assert result["status"] == "completed"
    assert result["preparation_seconds"] == pytest.approx(0.1)
    assert result["remaining_ns"]["black"] == 180_000_000_000 - 20_000_000
    assert result["remaining_ns"]["white"] == 180_000_000_000
    assert result["score_candidate"] == 1.0
    assert arena._read_receipt(tmp_path, folder / "receipt.json", "plan-hash") == result
    (folder / "events.jsonl.gz").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="changed"):
        arena._read_receipt(tmp_path, folder / "receipt.json", "plan-hash")


def test_stop_retains_confirmed_position_clocks_and_does_not_search(tmp_path, monkeypatch):
    plan = arena._plan(tmp_path, configuration(tmp_path))
    game = plan["games"][32]
    output = tmp_path / "local/arena"
    folder = output / "games/test/attempt-000"
    folder.mkdir(parents=True)
    (output / "STOP").touch()
    clock = SimulatedClock()
    monkeypatch.setattr(arena.time, "monotonic_ns", lambda: clock.now)
    monkeypatch.setattr(arena.time, "monotonic", lambda: clock.now / 1e9)
    result = arena._play_game(
        tmp_path, output, game, plan, "plan-hash", folder, helper(tmp_path, clock), None, 10_000
    )
    assert result["status"] == "stopped"
    assert result["moves"] == []
    assert result["remaining_ns"] == {"black": 180_000_000_000, "white": 180_000_000_000}
    assert result["score_candidate"] is None


def test_completed_games_resume_without_reruns_and_recovery_keeps_failed_attempt(
    tmp_path, monkeypatch
):
    config = configuration(tmp_path)
    (tmp_path / "scripts/compare_core_prototype.py").write_text("")
    output = tmp_path / "local/arena"
    calls = []

    def fake_game(root, _output, game, _plan, plan_sha, folder, _helper, _resumed, _deadline):
        calls.append(game["id"])
        trace = folder / "events.jsonl.gz"
        trace.write_bytes(gzip.compress(b"evidence"))
        failed = len(calls) == 1
        result = {
            **game,
            "plan_sha256": plan_sha,
            "status": "process_failure" if failed else "completed",
            "reason": "player_process_failure" if failed else "checkmate",
            "wall_seconds": 0.001,
            "score_candidate": None if failed else 1.0,
            "absolute_deadline_violations": 0,
            "trace": {"path": str(trace.relative_to(root)), "sha256": digest(trace)},
        }
        return arena._seal(folder / "receipt.json", result)

    monkeypatch.setattr(arena, "_play_game", fake_game)
    first = arena.run_arena(tmp_path, output, config)
    assert first["status"] == "complete"
    assert len(first["games"]) == 40
    assert len(first["attempts"]) == 41
    assert first["summary"]["adverse_attempts"] == 1
    assert not first["adoption_criteria_met"]
    calls.clear()
    resumed = arena.run_arena(tmp_path, output, config)
    assert not calls
    assert resumed["games"] == first["games"]
    assert resumed["attempts"] == first["attempts"]
    changed = dict(config, candidate_sha256="0" * 64)
    with pytest.raises(ValueError, match="changed"):
        arena.run_arena(tmp_path, output, changed)


def test_second_process_failure_closes_game_and_cannot_be_retried_again(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    (tmp_path / "scripts/compare_core_prototype.py").write_text("")
    output = tmp_path / "local/arena"
    calls = []

    def failed_game(root, _output, game, _plan, plan_sha, folder, _helper, _resumed, _deadline):
        calls.append(game["id"])
        trace = folder / "events.jsonl.gz"
        trace.write_bytes(gzip.compress(b"original failure evidence"))
        result = {
            **game,
            "plan_sha256": plan_sha,
            "status": "process_failure",
            "reason": "crash",
            "score_candidate": None,
            "wall_seconds": 0.001,
            "trace": {"path": str(trace.relative_to(root)), "sha256": digest(trace)},
        }
        return arena._seal(folder / "receipt.json", result)

    monkeypatch.setattr(arena, "_play_game", failed_game)
    first = arena.run_arena(tmp_path, output, config)
    assert first["status"] == "failed"
    assert len(calls) == 2
    assert len(first["attempts"]) == 2
    calls.clear()
    resumed = arena.run_arena(tmp_path, output, config)
    assert resumed["status"] == "failed"
    assert calls == []


def test_resumed_game_preserves_spent_clock_time(tmp_path, monkeypatch):
    plan = arena._plan(tmp_path, configuration(tmp_path))
    game = plan["games"][32]
    output = tmp_path / "local/arena"
    folder = output / "games/test/attempt-001"
    folder.mkdir(parents=True)
    prior = {
        "moves": [],
        "expected_sfen": START,
        "remaining_ns": {"black": 123_000_000_000, "white": 100_000_000_000},
        "receipt_sha256": "previous-confirmed-state",
    }
    clock = SimulatedClock()
    monkeypatch.setattr(arena.time, "monotonic_ns", lambda: clock.now)
    monkeypatch.setattr(arena.time, "monotonic", lambda: clock.now / 1e9)
    result = arena._play_game(
        tmp_path, output, game, plan, "plan-hash", folder, helper(tmp_path, clock), prior, 10_000
    )
    assert result["status"] == "completed"
    assert result["remaining_ns"] == {"black": 122_980_000_000, "white": 100_000_000_000}
    assert result["resumed_from"] == "previous-confirmed-state"
