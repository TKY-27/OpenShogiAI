"""Wire faults stay local; preserved C4 attempts and independent labels stay real."""

import hashlib
import json
import threading
from dataclasses import replace
from pathlib import Path

import pytest
from open_shogi_training import evaluator_run as runner
from open_shogi_training import r4_c4_data as data
from open_shogi_training.evaluator_data import atomic, encoded
from open_shogi_training.evaluator_ledger import Ledger, pack_result, unpack_result
from open_shogi_training.labeling import usi

from .helpers import make_fake_project


def engine_config(tmp_path, response):
    config, executable, _ = make_fake_project(tmp_path)
    executable.write_text(executable.read_text().replace("        good_search()", response))
    return replace(
        config,
        threads=1,
        multipv=1,
        binary_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
    )


def teacher(tmp_path, config):
    result = data.Teacher.__new__(data.Teacher)
    result.root, result.config = tmp_path, config
    result.ledger, result.engine = Ledger(tmp_path / "ledger"), None
    result.policy = dict(
        broad_depth=8, strong_depth=12, broad_nodes=200, strong_nodes=2000, maximum_worker_exits=3
    )
    return result


def candidate(move="7g7f", depth=8, bound=None):
    return usi.USICandidate(1, usi.USIScore("cp", 42), (move,), depth, 12, 20, bound)


def native(terminal="None", declaration=None):
    return {
        "sfen": "fixture b - 1",
        "terminal": terminal,
        "teacher_resign_eligible": terminal != "None",
        "teacher_declaration": declaration,
        "successors": [{"move": "7g7f", "sfen": "child b - 1", "terminal": "None"}]
        if terminal == "None"
        else [],
    }


@pytest.mark.parametrize("move", ["7g7f", "8h2b+", "P*5e"])
@pytest.mark.parametrize("suffix", ["", " ponder 3c3d"])
def test_normal_grammar_and_ponder_are_separate(move, suffix):
    result = usi._finish_search(
        f"bestmove {move}{suffix}", {1: candidate(move)}, expected_multipv=1, elapsed_ms=1
    )
    assert result.bestmove == move


@pytest.mark.parametrize(
    "line",
    [
        "bestmove",
        "bestmove ",
        "bestmove 0000",
        "bestmove (none)",
        "bestmove unknown",
        "bestmove 7g7f extra",
        "bestmove win extra",
    ],
)
def test_unknowns_are_neither_moves_nor_terminals(line):
    with pytest.raises(usi.USIProtocolError):
        usi._finish_search(line, {1: candidate()}, expected_multipv=1, elapsed_ms=1)


@pytest.mark.parametrize("role", ["player", "label", "branch"])
@pytest.mark.parametrize(
    "end", ["None", "Some(Checkmate { winner: White })", "Some(NoLegalMoves { winner: White })"]
)
def test_original_raw_resign_line_is_role_local(role, end):
    # Exact raw response reproduced with the original G01 request, not a cp/mate label.
    parsed = usi._finish_search("bestmove resign", {}, expected_multipv=1, elapsed_ms=1)
    signal = data.terminal_signal(parsed, native(end), role)
    assert not data.scalar(signal)
    outcome = data.player_outcome(signal, native(end))
    assert bool(outcome) == (role == "player")
    if end == "None":
        assert signal["validation"] == "teacher_resignation_not_mate_proof"
    assert unpack_result(pack_result(parsed)) == parsed


@pytest.mark.parametrize("role", ["player", "label", "branch"])
@pytest.mark.parametrize("valid", [True, False, None])
def test_win_needs_matching_native_rule_and_target(role, valid):
    declaration = (
        None
        if valid is None
        else {
            "rule": "csa_28_27",
            "side": "black",
            "minimum_camp_pieces": 10,
            "required_points": 28,
            "points": 28 if valid else None,
            "result": "win" if valid else "invalid_loss",
        }
    )
    state = native(declaration=declaration)
    result = usi.USITerminalResult("win", "bestmove win", 1)
    value = data.terminal_signal(result, state, role)
    end = data.player_outcome(value, state)
    if role != "player" or valid is None:
        assert end is None
    else:
        assert ("winner: Black" in end) == valid
        assert ("InvalidDeclaration" in end) != valid
    assert not data.scalar(value)


@pytest.mark.parametrize("depth,usable", [(8, True), (7, False)])
def test_resign_retains_only_independently_complete_info(tmp_path, depth, usable):
    config = engine_config(
        tmp_path,
        f"""        emit("info string bestmove win is text")
        emit("info depth {depth} seldepth 12 nodes 20 multipv 1 score cp 42 pv 7g7f")
        emit("bestmove resign")""",
    )
    t = teacher(tmp_path, config)
    try:
        key, value = t.observe("fixture b - 1", {"game": 1}, state=native())
        assert data.scalar(value) == usable
        assert data.player_outcome(value, native()) is None
        assert t.ledger.get(key)["status"] == "accepted"
        assert value["outcome"] == "resign"
    finally:
        t.close()
        t.ledger.close()


@pytest.mark.parametrize(
    "failure",
    [
        "emit('bestmove (none)')",
        "os._exit(7)",
        "sys.stdout.write('bestmove res'); sys.stdout.flush(); os._exit(0)",
        "emit('')",
    ],
)
def test_bad_task_deferred_next_real_pipe_task_succeeds_and_restart_keeps_budget(tmp_path, failure):
    config = engine_config(
        tmp_path,
        f"""        if 'BAD' in position:
            {failure}
            continue
        emit("info depth 8 seldepth 12 nodes 20 multipv 1 score cp 42 pv 7g7f")
        emit("bestmove 7g7f")""",
    )
    t = teacher(tmp_path, config)
    try:
        key, value = t.observe("BAD b - 1", {"game": 1})
        assert value is None
        saved = t.ledger.get(key)
        assert saved["status"] == "deferred" and len(saved["attempts"]) == 2
        assert saved["attempts"][0]["diagnostics"]["stdout_bytes_repr"]
        t.close()
        t.ledger.close()
        t = teacher(tmp_path, config)
        assert t.observe("BAD b - 1", {"game": 1})[1]["status"] == "missing"
        assert t.ledger.get(key) == saved
        good, value = t.observe("fixture b - 1", {"game": 2}, state=native())
        assert data.scalar(value) and t.ledger.get(good)["status"] == "accepted"
    finally:
        t.close()
        t.ledger.close()


def test_single_interrupted_attempt_has_only_one_slot_left(tmp_path):
    config = engine_config(tmp_path, "        emit('bestmove resign')")
    t = teacher(tmp_path, config)
    identity = {
        "game": 1,
        "sfen": "fixture b - 1",
        "depth": 8,
        "nodes": 200,
        "teacher": config.binary_sha256,
        "perspective": "side_to_move",
    }
    key, _ = t.ledger.task(identity)
    t.ledger.begin(key, 200)
    try:
        actual, value = t.observe(identity["sfen"], {"game": 1}, role="branch", state=native())
        assert actual == key and value["status"] == "terminal"
        attempts = t.ledger.get(key)["attempts"]
        assert len(attempts) == 2 and attempts[0]["outcome"] == "interrupted"
        assert attempts[1]["outcome"] == "accepted"
    finally:
        t.close()
        t.ledger.close()


def test_independent_unknown_responses_trip_health_without_fake_labels(tmp_path):
    config = engine_config(tmp_path, "        emit('bestmove unknown')")
    t = teacher(tmp_path, config)
    try:
        for game in range(3):
            assert t.observe(f"fixture{game} b - 1", {"game": game})[1] is None
        with pytest.raises(RuntimeError, match="independent tasks"):
            t.observe("fixture3 b - 1", {"game": 3})
        assert not t.ledger.db.execute("SELECT id FROM tasks WHERE status='accepted'").fetchall()
    finally:
        t.close()
        t.ledger.close()


def test_save_failure_and_wrong_position_do_not_become_missing(tmp_path, monkeypatch):
    config = engine_config(
        tmp_path,
        """        emit("info depth 8 seldepth 12 nodes 20 multipv 1 score cp 42 pv 7g7f")
        emit("bestmove 7g7f")""",
    )
    t = teacher(tmp_path, config)

    def cannot_save(*args, **kwargs):
        raise OSError("disk full")

    try:
        with pytest.raises(ValueError, match="mismatch"):
            t.observe("different b - 1", {"game": 0}, state=native())
        monkeypatch.setattr(t.ledger, "finish", cannot_save)
        with pytest.raises(OSError, match="disk full"):
            t.observe("fixture b - 1", {"game": 1}, state=native())
    finally:
        t.close()
        t.ledger.close()


def test_concurrent_request_is_rejected_before_wire_write(tmp_path):
    config, _, _ = make_fake_project(tmp_path)
    engine = usi.USIEngine(config, tmp_path)
    held, release = threading.Event(), threading.Event()

    def owner():
        with engine._request_lock:
            held.set()
            release.wait(3)

    thread = threading.Thread(target=owner)
    thread.start()
    assert held.wait(2)
    try:
        with pytest.raises(usi.USIConcurrentRequestError):
            engine.analyze("fixture b - 1")
        assert engine.pid is None and not engine._commands
    finally:
        release.set()
        thread.join()


def test_delayed_stop_and_stale_reply_never_reach_next_position(tmp_path):
    config = engine_config(
        tmp_path,
        """        if 'SLOW' in position:
            time.sleep(0.15)
            emit('bestmove resign')
            continue
        emit("info depth 8 seldepth 12 nodes 20 multipv 1 score cp 42 pv 7g7f")
        emit("bestmove 7g7f")""",
    )
    config = replace(config, timeouts=replace(config.timeouts, search_ms=50, stop_ms=300))
    with usi.USIEngine(config, tmp_path, allow_terminal_outcomes=True) as engine:
        first = engine.pid
        with pytest.raises(usi.USITimeoutError):
            engine.analyze("SLOW b - 1")
        assert engine.pid is None
        assert engine.analyze("fixture b - 1").bestmove == "7g7f"
        assert engine.pid != first and engine.search_diagnostics["process_generation"] == 2
        # A duplicate from a previous search arriving before readyok is ambiguous.
        engine._stdout._offer("bestmove resign")
        with pytest.raises(usi.USIProtocolError, match="isready"):
            engine.analyze("fixture b - 1")
        assert engine.pid is None


def test_monitor_ignores_heartbeat_and_retry_but_tracks_real_cursor(tmp_path):
    folder = tmp_path / "data/trajectories"
    ledger = Ledger(folder)
    key, _ = ledger.task({"game": 0, "sfen": "fixture b - 1"})
    before = data.progress(tmp_path)["signature"]
    ledger.begin(key, 200)
    atomic(tmp_path / "supervisor.log", b"alive")
    atomic(tmp_path / "monitor.jsonl", b"alive")
    running = data.progress(tmp_path)
    assert running["signature"] == before and running["active_task"]["id"] == key
    ledger.finish(key, "accepted", result={"status": "terminal", "outcome": "resign"})
    assert data.progress(tmp_path)["signature"] != before
    ledger.checkpoint(0, {"moves": ["7g7f"], "rows": []})
    assert data.progress(tmp_path)["cursor"]["plies"] == 1
    ledger.close()


def test_recovery_never_blanket_accepts_needs_astra(tmp_path, monkeypatch):
    current = {"status": "needs_astra", "reason": "stage_exit_1"}
    review = {
        "policy": runner.C4_TEACHER_POLICY,
        "original_state_sha256": hashlib.sha256(encoded(current)).hexdigest(),
    }
    assert runner._reviewed_c4_teacher_failure(tmp_path, current, review)
    assert not runner._reviewed_c4_teacher_failure(
        tmp_path, {**current, "reason": "disk_full"}, review
    )
    assert not runner._reviewed_c4_teacher_failure(tmp_path, current, None)


@pytest.mark.parametrize("reply", ["resign", "unknown"])
def test_generation_commits_after_branch_terminal_or_local_defer(tmp_path, monkeypatch, reply):
    config = engine_config(
        tmp_path,
        f"""        if position.startswith('child'):
            emit('bestmove {reply}')
            continue
        emit("info depth 8 seldepth 12 nodes 20 multipv 1 score cp 42 pv 7g7f")
        emit("bestmove 7g7f")""",
    )
    plan = json.loads((Path(__file__).parents[3] / "configs/evaluator-main.json").read_text())
    plan["runtime"] = {k: {"path": "fixture", "sha256": "a" * 64} for k in ("replay", "probe")}
    policy = plan["iteration"]["generation"]
    policy.update(max_plies=2, explore_stride=8, strong_roots_per_game=1, reply_plies=0)
    policy["teacher"].update(strong_depth=8, broad_depth=8)
    policy["starts"] = [{"moves": [], "family": "fixture", "group": "defense", "split": "train"}]

    class Replay:
        def __init__(self, *args):
            self.ply = 0

        def ask(self, *, reset=None, movement=None, **kwargs):
            self.ply = 0 if reset else self.ply + 1
            value = native("None" if self.ply < 2 else "Some(Checkmate { winner: Black })")
            value["sfen"] = ("child" if self.ply else "root") + " b - 1"
            return value

        def close(self):
            pass

    class Probe:
        def __init__(self, *args):
            self.nodes = 2048

        def search(self, *args):
            return {"best_move": "7g7f", "score": 0, "depth": 2, "nodes": 20, "iterations": []}

        def close(self):
            pass

    monkeypatch.setattr(data, "Replay", Replay)
    monkeypatch.setattr(data, "R3Probe", Probe)
    monkeypatch.setattr(data, "make_teacher_config", lambda *a: config)
    run = tmp_path / "local/run"
    atomic(run / "run.json", encoded(plan))
    folder = run / "data/trajectories"
    spec = {"id": "G01", "index": 1, "games": 2}
    actor = {"path": "fixture", "sha256": "a" * 64}
    result = data.generate(tmp_path, run, folder, plan, actor, spec)
    assert result["games"] == 2 and result["label_rows"] == 3
    assert len(list(folder.glob("*.receipt.json"))) == 2
    assert data.generate(tmp_path, run, folder, plan, actor, spec) == result
    ledger = Ledger(folder)
    tasks = ledger.db.execute("SELECT status,result,attempts FROM tasks").fetchall()
    assert any(json.loads(result)["status"] == "terminal" for _, result, _ in tasks) == (
        reply == "resign"
    )
    assert all(len(json.loads(attempts)) <= 2 for _, _, attempts in tasks)
    ledger.close()


def test_stop_failure_closes_ambiguous_channel(tmp_path):
    config, executable, _ = make_fake_project(tmp_path)
    executable.write_text(
        executable.read_text().replace(
            'elif line == "stop":\n        emit("bestmove 7g7f")',
            'elif line == "stop":\n        emit("checkmate nomate")',
        )
    )
    config = replace(config, binary_sha256=hashlib.sha256(executable.read_bytes()).hexdigest())
    with usi.USIEngine(config, tmp_path) as engine:
        first = engine.pid
        with pytest.raises(usi.USIProtocolError, match="unexpected stop output"):
            engine.stop()
        assert engine.pid is None
        assert engine.analyze("fixture b - 1").bestmove == "7g7f"
        assert engine.pid != first
